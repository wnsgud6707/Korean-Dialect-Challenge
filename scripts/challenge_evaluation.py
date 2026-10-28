"""Operator-only preparation and scoring; does not execute submitted code."""
import argparse
import json
from pathlib import Path
import random
import shutil

from baseline_data import ROOT, LABELS, digest, load_split, classification_metrics


def score_predictions(answers, predictions):
    expected = {r['sample_id']: r['label'] for r in answers}
    if not expected or len(expected) != len(answers) or any(v not in LABELS for v in expected.values()):
        raise ValueError('Invalid answer key')
    actual = {}
    for row in predictions:
        if set(row) != {'sample_id', 'prediction'}:
            raise ValueError('Prediction rows require exactly sample_id and prediction')
        if row['sample_id'] in actual or row['prediction'] not in LABELS:
            raise ValueError('Duplicate sample ID or invalid prediction label')
        actual[row['sample_id']] = row['prediction']
    if set(actual) != set(expected):
        raise ValueError('Prediction IDs must exactly match the evaluation set')
    ids = sorted(expected)
    return classification_metrics([LABELS.index(expected[i]) for i in ids],
                                  [LABELS.index(actual[i]) for i in ids])


def prepare(rows, source_hash, output, split):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    inputs_dir, private = output/'inputs', output/'private'
    (inputs_dir/'audio').mkdir(parents=True)
    private.mkdir()
    shuffled = list(rows)
    random.SystemRandom().shuffle(shuffled)
    inputs, answers = [], []
    for index, row in enumerate(shuffled):
        sid = f'eval_{index:06d}'
        relative = f'audio/{sid}.wav'
        destination = inputs_dir/relative
        shutil.copyfile(row['audio_path'],destination)
        if digest(destination) != row['wav_sha256']:
            raise ValueError('Evaluation audio copy checksum mismatch')
        inputs.append(dict(sample_id=sid, path=relative))
        answers.append(dict(sample_id=sid,label=row['region'],source_sample_id=row['sample_id'],wav_sha256=row['wav_sha256']))
    for path, rows_out in [(inputs_dir/'inputs.jsonl',inputs),(private/'answers.jsonl',answers)]:
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows_out),encoding='utf-8')
    receipt = dict(split=split,count=len(inputs),source_manifest_sha256=source_hash,
                   inputs_sha256=digest(inputs_dir/'inputs.jsonl'),answers_sha256=digest(private/'answers.jsonl'))
    (private/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare')
    p.add_argument('--version-dir',type=Path,default=ROOT/'data/benchmark_recording_v1')
    p.add_argument('--split',choices=['public_test','hidden_test'],default='public_test')
    p.add_argument('--organizer',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('score')
    s.add_argument('--evaluation',type=Path,required=True)
    s.add_argument('--predictions',type=Path,required=True)
    s.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        if args.split=='hidden_test' and not args.organizer:
            parser.error('Hidden preparation requires --organizer')
        rows,source_hash=load_split(args.version_dir,args.split,True)
        print(json.dumps(prepare(rows,source_hash,args.output,args.split)))
    else:
        root=args.evaluation
        receipt=json.loads((root/'private/receipt.json').read_text('utf-8'))
        for name, expected in [('private/answers.jsonl',receipt['answers_sha256']),('inputs/inputs.jsonl',receipt['inputs_sha256'])]:
            if digest(root/name)!=expected:
                raise ValueError('Evaluation metadata changed')
        answers=[json.loads(l) for l in (root/'private/answers.jsonl').read_text('utf-8').splitlines()]
        predictions=[json.loads(l) for l in args.predictions.read_text('utf-8').splitlines()]
        metrics=score_predictions(answers,predictions)
        result=dict(**metrics,split=receipt['split'],source_manifest_sha256=receipt['source_manifest_sha256'],
                    inputs_sha256=receipt['inputs_sha256'],predictions_sha256=digest(args.predictions))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as out:
            json.dump(result,out,ensure_ascii=False,indent=2)
        print(json.dumps(result))


if __name__=='__main__':
    main()
