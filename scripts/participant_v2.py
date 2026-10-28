"""Public v2 metadata, exactly eleven archives, deterministic local reconstruction."""
import argparse
from collections import defaultdict
import hashlib
import io
import json
from pathlib import Path
import re
import shutil
import wave
import zipfile
from aihub_labels import ROOT,sha256,validate_member
from participant_data import archive_path,download_archive,verify_archive,extract_one,verify_clips

KEYS={'gangwon':(118,{572593},572589),'gyeongsang':(119,{572703},572701),
      'jeolla':(120,{572657},572654),'jeju':(121,{572699,572695},572694),'chungcheong':(122,{572613},572612)}
PUBLIC=('train','validation','public_test')
COUNTS={'train':21222,'validation':3120,'public_test':3043}
FIELDS={'sample_id','region','split','path','frames','sample_rate','wav_sha256','pcm_sha256','datasetkey',
        'source_filekey','label_filekey','source_sha256','audio_member','audio_member_sha256','label_member',
        'label_sha256','utterance_index','start','end','preprocessing'}

def read_public(package):
    descriptor=json.loads((package/'dataset.json').read_text('utf-8'))
    if descriptor['version']!='recording-v2' or {e['name'] for e in descriptor['splits']}!=set(PUBLIC) or len(descriptor['splits'])!=3:
        raise ValueError('Expected only three public v2 splits')
    if descriptor['label2id']!={r:i for i,r in enumerate(['gyeongsang','jeolla','chungcheong','gangwon','jeju'])}:raise ValueError('Invalid label mapping')
    if sha256(package/'archives.json')!=descriptor['archive_inventory_sha256']:raise ValueError('Archive inventory changed')
    archives=json.loads((package/'archives.json').read_text('utf-8'))['archives']
    expected={(r,'source',d,k) for r,(d,ks,l) in KEYS.items() for k in ks}|{(r,'labels',d,l) for r,(d,ks,l) in KEYS.items()}
    if len(archives)!=11 or {(r['region'],r['kind'],r['datasetkey'],r['filekey']) for r in archives}!=expected:raise ValueError('Expected eleven fixed archives')
    for e in archives:
        validate_member(e['filename'])
        if not re.fullmatch('[0-9a-f]{64}',e['sha256']) or e['bytes']<=0 or e['transport_bytes']<=0:raise ValueError('Invalid archive lock')
    hashes={(e['datasetkey'],e['filekey']):e['sha256'] for e in archives}
    splits={};ids=set();groups=defaultdict(set)
    for entry in descriptor['splits']:
        split=entry['name']
        if entry['path']!=split+'.jsonl' or sha256(package/entry['path'])!=entry['sha256']:raise ValueError('Manifest lock changed')
        rows=[json.loads(l) for l in (package/entry['path']).read_text('utf-8').splitlines()]
        if len(rows)!=entry['rows'] or len(rows)!=COUNTS[split]:raise ValueError('Count differs')
        for r in rows:
            if set(r)!=FIELDS or r['split']!=split or r['region'] not in KEYS:raise ValueError('Unexpected public schema')
            sid=r['sample_id'];region=r['region'];d,ks,l=KEYS[region]
            if not re.fullmatch('[0-9a-f]{20}',sid) or sid in ids or r['path']!=f'clips/{region}/{sid}.wav':raise ValueError('Invalid public identity/path')
            ids.add(sid)
            if r['datasetkey']!=d or r['source_filekey'] not in ks or r['label_filekey']!=l:raise ValueError('Archive key mismatch')
            if r['source_sha256']!=hashes[d,r['source_filekey']]:raise ValueError('Source hash mismatch')
            for field in ('audio_member','label_member'):validate_member(r[field])
            for field in ('wav_sha256','pcm_sha256','audio_member_sha256','label_sha256'):
                if not re.fullmatch('[0-9a-f]{64}',r[field]):raise ValueError('Invalid checksum')
            if r['sample_rate']!=16000 or not 48000<=r['frames']<=160000 or not 0<=r['start']<r['end'] or round(r['end']*16000)-round(r['start']*16000)!=r['frames']:raise ValueError('Invalid audio span')
            if r['utterance_index']<0 or r['preprocessing']!='pcm16_mono_16k_polyphase_v1':raise ValueError('Unsupported extraction')
            for field,value in [('recording',f"{d}:{Path(r['audio_member']).stem}"),('source',r['audio_member_sha256']),('pcm',r['pcm_sha256'])]:groups[field,value].add(split)
        splits[split]=rows
    if any(len(v)>1 for v in groups.values()):raise ValueError('Cross-split overlap')
    return descriptor,archives,splits

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['plan','download','extract','verify','export-text'])
    p.add_argument('--package',type=Path,default=ROOT/'data/release_candidate_recording_v2/participant')
    p.add_argument('--downloads',type=Path,default=ROOT/'data/downloads')
    p.add_argument('--output',type=Path,default=ROOT/'data/participant_recording_v2');args=p.parse_args()
    descriptor,archives,splits=read_public(args.package);rows=[r for part in splits.values() for r in part]
    if args.command=='plan':
        print(json.dumps(dict(archives=len(archives),public_clips=len(rows),fresh_transfer_bytes=sum(e['transport_bytes'] for e in archives)),indent=2));return
    if args.command=='download':
        for entry in archives:download_archive(entry,args.downloads)
        print('Eleven fixed archives verified');return
    if args.command=='export-text':
        pairs={s:[] for s in ('train','validation')}
        for label in (e for e in archives if e['kind']=='labels'):
            path=archive_path(label,args.downloads);verify_archive(label,path)
            groups=defaultdict(list)
            for split in pairs:
                for r in splits[split]:
                    if r['region']==label['region']:groups[r['label_member']].append(r)
            with zipfile.ZipFile(path) as z:
                for member,items in sorted(groups.items()):
                    raw=z.read(member);digest=hashlib.sha256(raw).hexdigest();doc=json.loads(raw)
                    for r in items:
                        if digest!=r['label_sha256']:raise ValueError('Label hash differs')
                        u=doc['utterance'][r['utterance_index']]
                        if (u['start'],u['end'])!=(r['start'],r['end']):raise ValueError('Utterance differs')
                        pairs[r['split']].append(dict(sample_id=r['sample_id'],dialect=u['dialect_form'],standard=u['standard_form']))
        folder=args.output/'text';folder.mkdir(parents=True,exist_ok=True)
        for split,items in pairs.items():
            raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in sorted(items,key=lambda r:r['sample_id'])).encode('utf-8')
            path=folder/f'{split}.jsonl'
            if path.exists() and path.read_bytes()!=raw:raise ValueError('Existing text targets differ')
            if not path.exists():path.write_bytes(raw)
        print('Local train/validation text targets reconstructed; no test targets exported');return
    if args.command=='extract':
        args.output.mkdir(parents=True,exist_ok=True)
        for name in ('dataset.json','archives.json',*(s+'.jsonl' for s in PUBLIC)):
            target=args.output/name
            if target.exists() and sha256(target)!=sha256(args.package/name):raise ValueError('Output version differs')
            if not target.exists():shutil.copyfile(args.package/name,target)
        for e in archives:verify_archive(e,archive_path(e,args.downloads))
        for source in (e for e in archives if e['kind']=='source'):
            label=next(e for e in archives if e['region']==source['region'] and e['kind']=='labels')
            groups=defaultdict(list)
            for r in rows:
                if (r['datasetkey'],r['source_filekey'])==(source['datasetkey'],source['filekey']):groups[r['audio_member']].append(r)
            with zipfile.ZipFile(archive_path(source,args.downloads)) as audio,zipfile.ZipFile(archive_path(label,args.downloads)) as labels:
                for member,items in sorted(groups.items()):
                    pending=[]
                    for r in items:
                        path=args.output/r['path']
                        if path.exists():
                            if sha256(path)!=r['wav_sha256']:raise ValueError('Existing clip differs')
                        else:pending.append(r)
                    if not pending:continue
                    raw=audio.read(member);audio_hash=hashlib.sha256(raw).hexdigest();documents={}
                    with wave.open(io.BytesIO(raw),'rb') as wav:
                        for r in pending:
                            if r['label_member'] not in documents:
                                lab=labels.read(r['label_member']);documents[r['label_member']]=(hashlib.sha256(lab).hexdigest(),json.loads(lab))
                            label_hash,doc=documents[r['label_member']];u=doc['utterance'][r['utterance_index']]
                            if audio_hash!=r['audio_member_sha256'] or label_hash!=r['label_sha256'] or (u['start'],u['end'])!=(r['start'],r['end']):raise ValueError('Source/label mismatch')
                            data=extract_one(wav,r);path=args.output/r['path'];path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
            print(f'Extracted source {source["filekey"]}',flush=True)
    verify_clips(rows,args.output)
    report=dict(public_clips=len(rows),hidden_included=False,manifest_hashes={e['name']:e['sha256'] for e in descriptor['splits']})
    (args.output/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
