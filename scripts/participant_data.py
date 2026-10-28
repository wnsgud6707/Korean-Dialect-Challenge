"""Download the ten locked archives and extract exactly 9,001 public clips."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import shutil
import wave
import zipfile

from aihub_labels import ROOT, BASE, CurlResponse, api_key, catalog, rebuild_zip, sha256
from participant_manifest import read_public
from pilot_audio import read_clip


def archive_path(entry, downloads):
    return downloads / str(entry['datasetkey']) / str(entry['filekey']) / (entry['kind']+'.zip')


def verify_archive(entry, path):
    if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
        raise ValueError(f"Archive checksum/size mismatch: {entry['filekey']}")


def download_archive(entry, downloads):
    target = archive_path(entry, downloads)
    if target.exists():
        verify_archive(entry, target)
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    transport = target.parent / 'download.tar'
    required = entry['bytes'] + (0 if transport.exists() else entry['transport_bytes']) + 1024**3
    if shutil.disk_usage(target.parent).free < required:
        raise ValueError('Insufficient disk space for tar plus ZIP')
    if not transport.exists():
        partial = transport.with_suffix('.tar.partial')
        url = f"{BASE}/down/0.6/{entry['datasetkey']}.do?fileSn={entry['filekey']}"
        with CurlResponse(url, api_key(), max_bytes=entry['transport_bytes']) as response, partial.open('wb') as out:
            total = 0
            for block in iter(lambda: response.read(1024**2), b''):
                total += len(block)
                if total > entry['transport_bytes']:
                    raise ValueError('Download exceeded locked transfer size')
                out.write(block)
        if partial.stat().st_size != entry['transport_bytes']:
            raise ValueError('Incomplete transport')
        partial.replace(transport)
    temporary = target.with_suffix('.zip.partial')
    rebuild_zip(transport, temporary, entry['filename'])
    verify_archive(entry, temporary)
    temporary.replace(target)


def extract_one(wav, row):
    pcm = read_clip(wav, row['start'], row['end'])
    if len(pcm)//2 != row['frames'] or hashlib.sha256(pcm).hexdigest() != row['pcm_sha256']:
        raise ValueError('Extracted PCM differs from published clip')
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as out:
        out.setparams((1,2,16000,0,'NONE','not compressed'))
        out.writeframes(pcm)
    data = buffer.getvalue()
    if hashlib.sha256(data).hexdigest() != row['wav_sha256']:
        raise ValueError('Extracted WAV differs from published clip')
    return data


def verify_clips(rows, output):
    for row in rows:
        if sha256(output / row['path']) != row['wav_sha256']:
            raise ValueError(f"Output checksum mismatch: {row['sample_id']}")
    actual = {p.relative_to(output).as_posix() for p in (output / 'clips').rglob('*.wav')}
    if actual != {r['path'] for r in rows}:
        raise ValueError('Extra or missing WAV files in participant output')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan','download','extract','verify'])
    parser.add_argument('--package', type=Path, default=ROOT / 'data/release_candidate_recording_v1/participant')
    parser.add_argument('--downloads', type=Path, default=ROOT / 'data/downloads')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/participant_recording_v1')
    args = parser.parse_args()
    descriptor, archives, splits = read_public(args.package)
    rows = [r for group in splits.values() for r in group]
    if args.command == 'plan':
        print(json.dumps(dict(archives=len(archives), public_clips=len(rows),
            fresh_transfer_bytes=sum(e['transport_bytes'] for e in archives),
            missing_filekeys=[e['filekey'] for e in archives if not archive_path(e,args.downloads).exists()],
            note='Existing file presence only; content hashes verified during download/extract'), indent=2))
        return
    if args.command == 'download':
        missing = [e for e in archives if not archive_path(e,args.downloads).exists()]
        if missing:
            catalog([dict(e,display_size=str(e['bytes'])+' bytes') for e in missing])
        for entry in archives:
            download_archive(entry, args.downloads)
        print('Exactly ten archive locks verified; no other downloads requested')
        return
    if args.command == 'extract':
        args.output.mkdir(parents=True, exist_ok=True)
        for name in ('dataset.json','archives.json','train.jsonl','validation.jsonl','public_test.jsonl'):
            target, source = args.output / name, args.package / name
            if target.exists() and sha256(target) != sha256(source):
                raise ValueError('Output belongs to a different dataset version')
            if not target.exists():
                shutil.copyfile(source, target)
        for entry in archives:
            print(f"Verifying archive {entry['filekey']}", flush=True)
            verify_archive(entry, archive_path(entry,args.downloads))
        for source in (e for e in archives if e['kind']=='source'):
            label = next(e for e in archives if e['region']==source['region'] and e['kind']=='labels')
            grouped = defaultdict(list)
            for row in rows:
                if row['region']==source['region']:
                    grouped[row['audio_member']].append(row)
            count = 0
            with zipfile.ZipFile(archive_path(source,args.downloads)) as audio_zip, zipfile.ZipFile(archive_path(label,args.downloads)) as label_zip:
                for member, items in sorted(grouped.items()):
                    pending = []
                    for row in items:
                        path = args.output / row['path']
                        if path.exists():
                            if sha256(path) != row['wav_sha256']:
                                raise ValueError('Existing clip changed; refusing to overwrite')
                        else:
                            pending.append(row)
                    if pending:
                        audio = audio_zip.read(member)
                        audio_hash = hashlib.sha256(audio).hexdigest()
                        documents = {}
                        with wave.open(io.BytesIO(audio),'rb') as wav:
                            for row in pending:
                                if row['label_member'] not in documents:
                                    raw_label = label_zip.read(row['label_member'])
                                    documents[row['label_member']] = (hashlib.sha256(raw_label).hexdigest(),json.loads(raw_label))
                                label_hash, document = documents[row['label_member']]
                                if audio_hash != row['audio_member_sha256'] or label_hash != row['label_sha256']:
                                    raise ValueError('Source WAV or label hash mismatch')
                                utterance = document['utterance'][row['utterance_index']]
                                if (utterance['start'],utterance['end']) != (row['start'],row['end']):
                                    raise ValueError('Label timestamp mismatch')
                                data = extract_one(wav,row)
                                path = args.output / row['path']
                                path.parent.mkdir(parents=True,exist_ok=True)
                                temporary = path.with_suffix('.wav.partial')
                                temporary.write_bytes(data)
                                temporary.replace(path)
                    count += len(items)
                print(f"{source['region']}: {count} public clips verified", flush=True)
    verify_clips(rows,args.output)
    report = dict(status='verified',clips=len(rows),split_counts={s:len(r) for s,r in splits.items()},
                  hidden_clips_included=False,manifest_hashes={e['name']:e['sha256'] for e in descriptor['splits']})
    (args.output / 'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
