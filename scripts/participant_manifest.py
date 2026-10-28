"""Read only public release metadata; never consult organizer files."""
from collections import Counter, defaultdict
import json
import re
from pathlib import Path
from aihub_labels import sha256, validate_member

SPLITS = {'train': 7001, 'validation': 1000, 'public_test': 1000}
LABELS = ['gyeongsang', 'jeolla', 'chungcheong', 'gangwon', 'jeju']
KEYS = {'gangwon': (118,572593,572589), 'gyeongsang': (119,572703,572701),
        'jeolla': (120,572657,572654), 'jeju': (121,572699,572694),
        'chungcheong': (122,572613,572612)}


def read_public(package):
    base = Path(package).resolve()
    descriptor = json.loads((base / 'dataset.json').read_text('utf-8'))
    if descriptor['version'] != 'recording-v1' or descriptor['label2id'] != dict(zip(LABELS, range(5))):
        raise ValueError('Unsupported dataset version or label mapping')
    entries = descriptor['splits']
    if Counter(e['name'] for e in entries) != Counter({s:1 for s in SPLITS}):
        raise ValueError('Only the three public splits may be present')
    if sha256(base / 'archives.json') != descriptor['archive_inventory_sha256']:
        raise ValueError('Archive inventory checksum mismatch')
    archive_doc = json.loads((base / 'archives.json').read_text('utf-8'))
    archives = archive_doc['archives']
    expected = {(region, kind, keys[0], keys[index]) for region, keys in KEYS.items()
                for kind, index in [('source',1), ('labels',2)]}
    actual = {(e['region'], e['kind'], e['datasetkey'], e['filekey']) for e in archives}
    if len(archives) != 10 or actual != expected:
        raise ValueError('Expected exactly the ten published archive keys')
    for e in archives:
        validate_member(e['filename'])
        if not re.fullmatch('[0-9a-f]{64}', e['sha256']) or e['bytes'] <= 0 or e['transport_bytes'] <= 0:
            raise ValueError('Invalid archive lock')
    if sum(e['transport_bytes'] for e in archives) != archive_doc['expected_fresh_transport_bytes']:
        raise ValueError('Transfer total mismatch')
    source_hashes = {e['region']:e['sha256'] for e in archives if e['kind']=='source'}
    splits, ids, groups = {}, set(), defaultdict(set)
    for entry in entries:
        split = entry['name']
        if entry['path'] != split + '.jsonl' or entry['rows'] != SPLITS[split]:
            raise ValueError('Invalid public manifest path/count')
        path = base / entry['path']
        if sha256(path) != entry['sha256']:
            raise ValueError('Public manifest checksum mismatch')
        rows = [json.loads(line) for line in path.read_text('utf-8').splitlines()]
        if len(rows) != entry['rows']:
            raise ValueError('Wrong public row count')
        for row in rows:
            region, sid = row['region'], row['sample_id']
            if region not in KEYS or not re.fullmatch('[0-9a-f]{20}', sid) or sid in ids:
                raise ValueError('Invalid or duplicate sample ID/region')
            ids.add(sid)
            if row['split'] != split or row['path'] != f'clips/{region}/{sid}.wav':
                raise ValueError('Wrong split or unsafe output path')
            if (row['datasetkey'], row['source_filekey'], row['label_filekey']) != KEYS[region]:
                raise ValueError('Sample archive keys mismatch')
            if row['source_sha256'] != source_hashes[region]:
                raise ValueError('Sample source hash mismatch')
            for field in ('audio_member', 'label_member'):
                validate_member(row[field])
            for field in ('wav_sha256', 'pcm_sha256', 'audio_member_sha256', 'label_sha256'):
                if not re.fullmatch('[0-9a-f]{64}', row[field]):
                    raise ValueError('Invalid sample hash')
            if row['sample_rate'] != 16000 or not 48000 <= row['frames'] <= 160000:
                raise ValueError('Wrong audio format/length')
            if not (0 <= row['start'] < row['end']) or round(row['end']*16000)-round(row['start']*16000) != row['frames']:
                raise ValueError('Invalid sample timestamps')
            if row['utterance_index'] < 0 or row['preprocessing'] != 'pcm16_mono_16k_polyphase_v1':
                raise ValueError('Unsupported extraction recipe')
            row['recording_key'] = f"{row['datasetkey']}:{Path(row['audio_member']).stem}"
            for field in ('recording_key', 'audio_member_sha256', 'pcm_sha256'):
                groups[(field, row[field])].add(split)
        target = {label: (1401 if label=='jeju' else 1400) if split=='train' else 200 for label in LABELS}
        if Counter(r['region'] for r in rows) != target:
            raise ValueError('Class count mismatch')
        splits[split] = rows
    if any(len(s) != 1 for s in groups.values()):
        raise ValueError('Recording or audio content crosses public splits')
    return descriptor, archives, splits
