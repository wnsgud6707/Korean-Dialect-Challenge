"""Operator-only joint dialect/standard-text evaluation; never runs submission code."""
import argparse
import json
from pathlib import Path
import sys
from baseline_data import ROOT,LABELS,digest,load_split
from challenge_evaluation import prepare,score_predictions
sys.path.insert(0,str(ROOT/'experiments/standardization_v1'))
from text_metrics import error_rates

def score_joint(answers,predictions):
    if any(set(r)!={'sample_id','prediction','standard_text'} or not isinstance(r['standard_text'],str) for r in predictions):
        raise ValueError('Require sample_id, prediction, standard_text only')
    classification=score_predictions(answers,[dict(sample_id=r['sample_id'],prediction=r['prediction']) for r in predictions])
    actual={r['sample_id']:r for r in predictions};ordered=sorted(answers,key=lambda r:r['sample_id'])
    text=error_rates([r['standard'] for r in ordered],[actual[r['sample_id']]['standard_text'] for r in ordered])
    per_region={region:error_rates([r['standard'] for r in ordered if r['label']==region],
        [actual[r['sample_id']]['standard_text'] for r in ordered if r['label']==region]) for region in LABELS if any(r['label']==region for r in ordered)}
    return dict(classification=classification,standardization=text,standardization_per_region=per_region)

def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('prepare');p.add_argument('--split',choices=['public_test','hidden_test'],default='public_test');p.add_argument('--organizer',action='store_true');p.add_argument('--output',type=Path,required=True)
    s=sub.add_parser('score');s.add_argument('--evaluation',type=Path,required=True);s.add_argument('--predictions',type=Path,required=True);s.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='prepare':
        if args.split=='hidden_test' and not args.organizer:parser.error('Hidden preparation requires --organizer')
        rows,source_hash=load_split(ROOT/'data/benchmark_recording_v2',args.split,True)
        selection=ROOT/'data/expansion_v2/selection';lock=json.loads((selection/'report.json').read_text('utf-8'))
        source=selection/f'{args.split}.jsonl'
        if digest(source)!=lock['manifests'][args.split]:raise ValueError('Text selection changed')
        targets={r['sample_id']:r for r in map(json.loads,source.read_text('utf-8').splitlines())}
        if set(targets)!=set(r['sample_id'] for r in rows):raise ValueError('Text/audio IDs differ')
        receipt=prepare(rows,source_hash,args.output,args.split)
        path=args.output/'private/answers.jsonl';answers=[json.loads(l) for l in path.read_text('utf-8').splitlines()]
        for r in answers:r['standard']=targets[r['source_sample_id']]['standard']
        path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in answers),encoding='utf-8')
        receipt.update(task='joint-recording-v2',answers_sha256=digest(path),text_source_sha256=digest(source))
        (args.output/'private/receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8');print(json.dumps(receipt))
    else:
        root=args.evaluation;receipt=json.loads((root/'private/receipt.json').read_text('utf-8'))
        if receipt['task']!='joint-recording-v2':raise ValueError('Wrong evaluation task')
        if digest(root/'private/answers.jsonl')!=receipt['answers_sha256'] or digest(root/'inputs/inputs.jsonl')!=receipt['inputs_sha256']:raise ValueError('Evaluation metadata changed')
        answers=[json.loads(l) for l in (root/'private/answers.jsonl').read_text('utf-8').splitlines()]
        predictions=[json.loads(l) for l in args.predictions.read_text('utf-8').splitlines()]
        result=dict(**score_joint(answers,predictions),split=receipt['split'],predictions_sha256=digest(args.predictions),receipt=receipt)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open('x',encoding='utf-8') as out:json.dump(result,out,ensure_ascii=False,indent=2)
        print(json.dumps({k:v for k,v in result.items() if k!='receipt'}))

if __name__=='__main__':main()
