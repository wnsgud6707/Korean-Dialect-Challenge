"""Reference inference interface: model + unlabeled inputs -> JSONL predictions."""
import argparse
import json
from pathlib import Path
import wave

import baseline
from baseline_data import LABELS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=2)
    args=parser.parse_args()
    if args.batch_size<1 or args.output.exists():
        raise ValueError('Use a positive batch size and a new output file')
    base=args.inputs.resolve().parent
    rows=[json.loads(l) for l in args.inputs.read_text('utf-8').splitlines()]
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('Empty inputs or duplicate IDs')
    for row in rows:
        if set(row)!={'sample_id','path'}:
            raise ValueError('Only unlabeled sample_id/path inputs accepted')
        audio=(base/row['path']).resolve()
        if not audio.is_relative_to(base):
            raise ValueError('Audio path escapes input directory')
        with wave.open(str(audio),'rb') as wav:
            if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth())!=(16000,1,2):
                raise ValueError('Expected 16kHz mono PCM16')
        row['audio_path']=str(audio)
    baseline.dependencies()
    torch=baseline.torch
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=baseline.Wav2Vec2BertForSequenceClassification.from_pretrained(args.model,local_files_only=True).to(device)
    if model.config.label2id!=dict(zip(LABELS,range(5))):
        raise ValueError('Model label mapping differs from challenge mapping')
    extractor=baseline.AutoFeatureExtractor.from_pretrained(args.model,local_files_only=True)
    model.eval()
    predictions=[]
    with torch.inference_mode():
        for start in range(0,len(rows),args.batch_size):
            batch=rows[start:start+args.batch_size]
            ids=model(**baseline.batch_inputs(batch,extractor,device)).logits.argmax(-1).cpu().tolist()
            predictions.extend(dict(sample_id=r['sample_id'],prediction=LABELS[i]) for r,i in zip(batch,ids))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as out:
        out.write(''.join(json.dumps(r)+'\n' for r in predictions))
    print(f'Wrote {len(predictions)} predictions; no labels were provided')


if __name__=='__main__':
    main()
