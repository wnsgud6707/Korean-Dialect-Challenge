"""Reference submission: inputs only, region and standard text, measured batch-1 latency."""
import argparse
import json
from pathlib import Path
import sys
import time
import wave
import numpy as np
import torch
from transformers import WhisperProcessor,WhisperForConditionalGeneration,AutoFeatureExtractor,Wav2Vec2BertForSequenceClassification
from standardize import ROOT,edit
sys.path.insert(0,str(ROOT/'scripts'))
from baseline_data import LABELS,digest

def synchronize():
    if torch.cuda.is_available():torch.cuda.synchronize()

class JointModel:
    def __init__(self,classifier,asr,editor):
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.dtype=torch.float16 if self.device=='cuda' else torch.float32
        self.classifier=Wav2Vec2BertForSequenceClassification.from_pretrained(classifier,local_files_only=True).to(self.device).eval()
        if self.classifier.config.label2id!={label:i for i,label in enumerate(LABELS)}:raise ValueError('Wrong classifier label order')
        self.features=AutoFeatureExtractor.from_pretrained(classifier,local_files_only=True)
        self.processor=WhisperProcessor.from_pretrained(asr,local_files_only=True)
        self.asr=WhisperForConditionalGeneration.from_pretrained(asr,local_files_only=True,torch_dtype=self.dtype).to(self.device).eval()
        self.asr.config.forced_decoder_ids=None;self.asr.generation_config.forced_decoder_ids=None
        self.rules=json.loads((editor/'rules.json').read_text('utf-8'))
        self.config=json.loads((editor/'text_evaluation.json').read_text('utf-8'))['selected']

    def predict(self,path):
        synchronize();start=time.perf_counter()
        with wave.open(str(path),'rb') as w:
            if (w.getframerate(),w.getnchannels(),w.getsampwidth())!=(16000,1,2) or not 48000<=w.getnframes()<=160000:raise ValueError('Expected benchmark 3–10 second 16kHz mono PCM16')
            seconds=w.getnframes()/16000;audio=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.float32)/32768
        with torch.inference_mode():
            inputs=self.features([audio],sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
            logits=self.classifier(**{k:v.to(self.device) for k,v in inputs.items()}).logits
            region=LABELS[logits.argmax(-1).item()]
            features=self.processor(audio,sampling_rate=16000,return_tensors='pt',return_attention_mask=True)
            ids=self.asr.generate(features.input_features.to(self.device,dtype=self.dtype),attention_mask=features.attention_mask.to(self.device),
                                  language='ko',task='transcribe',max_new_tokens=160,do_sample=False)
        transcript=self.processor.batch_decode(ids,skip_special_tokens=True)[0].strip()
        standard=edit(transcript,self.rules,self.config['support'],self.config['confidence'])
        synchronize();elapsed=time.perf_counter()-start
        return dict(prediction=region,standard_text=standard),elapsed,seconds

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--classifier',type=Path,required=True);p.add_argument('--asr',type=Path,required=True);p.add_argument('--editor',type=Path,required=True);args=p.parse_args()
    if args.output.exists() or args.output.with_suffix('.timing.json').exists():raise ValueError('Choose new output paths')
    rows=[json.loads(l) for l in args.inputs.read_text('utf-8').splitlines()];base=args.inputs.parent.resolve()
    if not rows or len({r['sample_id'] for r in rows})!=len(rows):raise ValueError('Empty/duplicate inputs')
    for r in rows:
        if set(r)!={'sample_id','path'}:raise ValueError('Inputs must not contain labels')
        path=(base/r['path']).resolve()
        if not path.is_relative_to(base) or not path.is_file():raise ValueError('Invalid audio path')
    torch.manual_seed(42);start=time.perf_counter();engine=JointModel(args.classifier,args.asr,args.editor);synchronize();load_seconds=time.perf_counter()-start
    engine.predict(base/rows[0]['path'])
    args.output.parent.mkdir(parents=True,exist_ok=True);temporary=args.output.with_suffix('.partial')
    times=[];durations=[]
    with temporary.open('w',encoding='utf-8') as stream:
        for i,r in enumerate(rows,1):
            prediction,elapsed,seconds=engine.predict(base/r['path']);times.append(elapsed);durations.append(seconds)
            stream.write(json.dumps(dict(sample_id=r['sample_id'],**prediction),ensure_ascii=False)+'\n')
            if i%100==0:print(f'Joint predictions {i}/{len(rows)}',flush=True)
    temporary.replace(args.output)
    timing=dict(count=len(rows),batch_size=1,seed=42,warmup_clips=1,model_load_seconds=load_seconds,device=engine.device,
        gpu=torch.cuda.get_device_name(0) if engine.device=='cuda' else None,torch_version=torch.__version__,
        mean_seconds=float(np.mean(times)),p50_seconds=float(np.percentile(times,50)),p95_seconds=float(np.percentile(times,95)),
        aggregate_rtf=sum(times)/sum(durations),total_audio_seconds=sum(durations),total_processing_seconds=sum(times),
        scope='WAV read + feature extraction + classification + ASR + phrase edit; excludes model load, warmup and prediction JSON writes',
        inputs_sha256=digest(args.inputs),predictions_sha256=digest(args.output),rules_sha256=digest(args.editor/'rules.json'),
        classifier_sha256=digest(args.classifier/'model.safetensors'),asr_sha256=digest(args.asr/'model.safetensors'))
    args.output.with_suffix('.timing.json').write_text(json.dumps(timing,indent=2),encoding='utf-8');print(json.dumps(timing))

if __name__=='__main__':main()
