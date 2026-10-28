"""Fixed-manifest validation and metrics; no training dependencies required."""
import hashlib
import json
from pathlib import Path
import wave

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['gyeongsang', 'jeolla', 'chungcheong', 'gangwon', 'jeju']


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_split(version_dir, split, verify_audio=False):
    if split not in ('train', 'validation', 'public_test', 'hidden_test'):
        raise ValueError('Unknown split')
    base = Path(version_dir).resolve()
    if (base / 'dataset.json').exists():
        if split == 'hidden_test':
            raise ValueError('Hidden test is not part of the participant dataset')
        descriptor_version=json.loads((base/'dataset.json').read_text('utf-8'))['version']
        if descriptor_version=='recording-v2':
            from participant_v2 import read_public
        else:
            from participant_manifest import read_public
        descriptor, _, splits = read_public(base)
        entry = next(e for e in descriptor['splits'] if e['name']==split)
        rows = splits[split]
        if descriptor_version=='recording-v2':
            for row in rows:
                row['recording_key']=f"{row['datasetkey']}:{Path(row['audio_member']).stem}"
        audio_root = base
    else:
        report = json.loads((base / 'organizer/split_report.json').read_text('utf-8'))
        relative = f"{'organizer' if split == 'hidden_test' else 'public'}/{split}.jsonl"
        entry = next(e for e in report['manifests'] if e['path'] == relative)
        path = base / relative
        if digest(path) != entry['sha256']:
            raise ValueError('Frozen manifest checksum mismatch')
        rows = [json.loads(line) for line in path.read_text('utf-8').splitlines()]
        if len(rows) != entry['rows'] or len({r['sample_id'] for r in rows}) != len(rows):
            raise ValueError('Manifest count or duplicate ID error')
        audio_root = (base / report['audio_root_relative_to_version_dir']).resolve()
    for row in rows:
        audio = (audio_root / row['path']).resolve()
        if not audio.is_relative_to(audio_root) or not audio.is_file():
            raise ValueError('Invalid or missing audio path')
        if row['split'] != split or row['region'] not in LABELS:
            raise ValueError('Split or label mismatch')
        if verify_audio:
            if digest(audio) != row['wav_sha256']:
                raise ValueError(f"Audio checksum mismatch: {row['sample_id']}")
            with wave.open(str(audio), 'rb') as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getnframes()) != (16000, 1, 2, row['frames']):
                    raise ValueError('Audio format mismatch')
                if not 48000 <= wav.getnframes() <= 160000:
                    raise ValueError('Audio duration outside policy')
        row['audio_path'] = str(audio)
    return rows, entry['sha256']


def classification_metrics(truth, predictions):
    if len(truth) != len(predictions) or not truth:
        raise ValueError('Need equally sized, nonempty labels and predictions')
    matrix = [[0] * 5 for _ in range(5)]
    for actual, predicted in zip(truth, predictions):
        if actual not in range(5) or predicted not in range(5):
            raise ValueError('Label ID outside 0..4')
        matrix[actual][predicted] += 1
    f1 = []
    for i in range(5):
        denom = sum(matrix[i]) + sum(row[i] for row in matrix)
        f1.append(2 * matrix[i][i] / denom if denom else 0.0)
    return dict(macro_f1=sum(f1) / 5,
                accuracy=sum(matrix[i][i] for i in range(5)) / len(truth),
                per_class_f1=dict(zip(LABELS, f1)), confusion_matrix=matrix,
                label_order=LABELS, confusion_matrix_axes='rows=true, columns=predicted',
                count=len(truth))
