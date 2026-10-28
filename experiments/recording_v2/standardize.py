"""Train phrase edits on recording-v2 train; compare local ASR on validation only."""
import argparse
from collections import Counter,defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys
import time
import wave

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiments/standardization_v1'))
from improve import edit,evaluate
from text_metrics import normalize,error_rates
OUT=ROOT/'data/standardization_recording_v2'
PARTICIPANT=False

def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')

def paired(split):
    if split not in ('train','validation'):raise ValueError('Development uses train/validation only')
    if PARTICIPANT:
        sys.path.insert(0,str(ROOT/'scripts'))
        from participant_v2 import read_public
        base=ROOT/'data/participant_recording_v2'
        rows=read_public(base)[2][split]
        targets={r['sample_id']:r for r in map(json.loads,(base/'text'/f'{split}.jsonl').read_text('utf-8').splitlines())}
        if set(targets)!=set(r['sample_id'] for r in rows):raise ValueError('Text target IDs differ')
        return [dict(r,dialect=targets[r['sample_id']]['dialect'],standard=targets[r['sample_id']]['standard'],audio_path=str(base/r['path'])) for r in rows]
    base=ROOT/'data/expansion_v2/selection';path=base/f'{split}.jsonl'
    report=json.loads((base/'report.json').read_text('utf-8'))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=report['manifests'][split]:raise ValueError('Selection changed')
    return [json.loads(l) for l in path.read_text('utf-8').splitlines()]

def train():
    OUT.mkdir(parents=True,exist_ok=False)
    rows=paired('train');candidates=defaultdict(Counter)
    for r in rows:
        a,b=normalize(r['dialect']).split(),normalize(r['standard']).split()
        for tag,i,j,k,l in SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
            if tag!='replace' or not 1<=j-i<=3 or not 1<=l-k<=3:continue
            source,target=tuple(a[i:j]),tuple(b[k:l])
            if any(not all('\uac00'<=c<='\ud7a3' for c in t) for t in source+target):continue
            candidates[source][target]+=1
    occurrences=Counter()
    for r in rows:
        tokens=normalize(r['dialect']).split()
        for i in range(len(tokens)):
            for n in (1,2,3):
                key=tuple(tokens[i:i+n])
                if len(key)==n and key in candidates:occurrences[key]+=1
    rules=[]
    for source,targets in sorted(candidates.items()):
        target,count=targets.most_common(1)[0]
        rules.append(dict(source=list(source),target=list(target),support=count,confidence=count/occurrences[source]))
    write(OUT/'rules.json',rules)
    val=paired('validation');grid=[]
    for support in (2,3,5):
        for confidence in (.8,.9,1.0):
            predictions=[edit(r['dialect'],rules,support,confidence) for r in val]
            grid.append(dict(support=support,confidence=confidence,metrics=evaluate(val,predictions)))
    selected=min(grid,key=lambda r:r['metrics']['all']['cer'])
    write(OUT/'text_evaluation.json',dict(dataset='recording-v2',train_count=len(rows),validation_count=len(val),test_used=False,
        unchanged_baseline=evaluate(val,[r['dialect'] for r in val]),candidates=grid,selected=selected))
    print(json.dumps(dict(train=len(rows),validation=len(val),rules=len(rules),selected=selected)),flush=True)

def asr(model_key):
    import numpy as np
    import torch
    from transformers import WhisperProcessor,WhisperForConditionalGeneration
    rows=paired('validation');rules=json.loads((OUT/'rules.json').read_text('utf-8'))
    chosen=json.loads((OUT/'text_evaluation.json').read_text('utf-8'))['selected']
    model_path=ROOT/'data/standardization_models'/model_key
    cache=OUT/f'{model_key}_validation.jsonl'
    metadata_path=OUT/f'{model_key}_validation_lock.json'
    metadata=dict(model_revision=json.loads((model_path/'receipt.json').read_text())['revision'],
        rules_sha256=hashlib.sha256((OUT/'rules.json').read_bytes()).hexdigest(),support=chosen['support'],confidence=chosen['confidence'],
        validation_sha256=hashlib.sha256(json.dumps([(r['sample_id'],r['label_sha256'],r['dialect'],r['standard']) for r in rows],ensure_ascii=False).encode()).hexdigest(),
        language='ko',task='transcribe',max_new_tokens=160,do_sample=False,batch_size=4,seed=42)
    if metadata_path.exists():
        if json.loads(metadata_path.read_text('utf-8'))!=metadata:raise ValueError('Cached inference belongs to another configuration')
    elif cache.exists():raise ValueError('Cache without provenance lock')
    else:write(metadata_path,metadata)
    torch.manual_seed(42);device='cuda' if torch.cuda.is_available() else 'cpu';dtype=torch.float16 if device=='cuda' else torch.float32
    processor=WhisperProcessor.from_pretrained(model_path,local_files_only=True)
    model=WhisperForConditionalGeneration.from_pretrained(model_path,local_files_only=True,torch_dtype=dtype).to(device).eval()
    model.config.forced_decoder_ids=None;model.generation_config.forced_decoder_ids=None
    done={}
    if cache.exists():
        for line in cache.read_text('utf-8').splitlines():
            r=json.loads(line);done[r['sample_id']]=r
    remaining=[r for r in rows if r['sample_id'] not in done]
    start=time.perf_counter()
    with cache.open('a',encoding='utf-8') as stream:
        for i in range(0,len(remaining),4):
            part=remaining[i:i+4];audios=[]
            for r in part:
                path=r.get('audio_path',str(ROOT/f'data/expansion_v2/audio/{r["region"]}/{r["sample_id"]}.wav'))
                with wave.open(path,'rb') as w:
                    if (w.getframerate(),w.getnchannels(),w.getsampwidth())!=(16000,1,2):raise ValueError('Audio format differs')
                    audios.append(np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768)
            inputs=processor(audios,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
            with torch.inference_mode():
                ids=model.generate(inputs.input_features.to(device,dtype=dtype),attention_mask=inputs.attention_mask.to(device),
                    language='ko',task='transcribe',max_new_tokens=160,do_sample=False)
            texts=processor.batch_decode(ids,skip_special_tokens=True)
            for r,text in zip(part,texts):
                result=dict(sample_id=r['sample_id'],transcript=text.strip(),standard_candidate=edit(text.strip(),rules,chosen['support'],chosen['confidence']))
                stream.write(json.dumps(result,ensure_ascii=False)+'\n');done[r['sample_id']]=result
            stream.flush()
            if (i+4)%100==0:print(f'{model_key}: {len(done)}/{len(rows)}',flush=True)
    if set(done)!=set(r['sample_id'] for r in rows):raise ValueError('Prediction IDs differ')
    texts=[done[r['sample_id']]['transcript'] for r in rows];converted=[done[r['sample_id']]['standard_candidate'] for r in rows]
    report=dict(model=model_key,model_revision=json.loads((model_path/'receipt.json').read_text())['revision'],
        dataset='recording-v2',split='validation',test_used=False,batch_size=4,device=device,
        asr_vs_dialect=error_rates([r['dialect'] for r in rows],texts),asr_vs_standard=evaluate(rows,texts),
        converted_vs_standard=evaluate(rows,converted),elapsed_this_run_seconds=time.perf_counter()-start,
        timing_note='Batch throughput including decoding and local writes; not single-request latency; excludes model load',
        prediction_sha256=hashlib.sha256(cache.read_bytes()).hexdigest())
    write(OUT/f'{model_key}_metrics.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='asr_vs_standard'}),flush=True)

def main():
    global OUT,PARTICIPANT
    p=argparse.ArgumentParser();p.add_argument('command',choices=['train','asr']);p.add_argument('--model',choices=['asr','asr_turbo'],default='asr')
    p.add_argument('--participant',action='store_true');args=p.parse_args()
    PARTICIPANT=args.participant
    if PARTICIPANT:OUT=ROOT/'data/standardization_participant_recording_v2'
    if args.command=='train':train()
    else:asr(args.model)

if __name__=='__main__':main()
