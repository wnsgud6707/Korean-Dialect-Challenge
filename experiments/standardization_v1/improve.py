"""Train-only phrase editor and complete validation evaluation; never reads test labels."""
import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from baseline_data import load_split
from text_metrics import normalize,error_rates,apply_lexicon
OUT=ROOT/'data/standardization_v2'

def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2),encoding='utf-8')

def prepare():
    OUT.mkdir(exist_ok=False)
    for split in ('train','validation'):
        rows,lock=load_split(ROOT/'data/participant_recording_v1',split)
        groups=defaultdict(list)
        for r in rows:groups[r['datasetkey'],r['label_filekey']].append(r)
        pairs=[]
        for (dataset,filekey),group in groups.items():
            with zipfile.ZipFile(ROOT/f'data/downloads/{dataset}/{filekey}/labels.zip') as z:
                cache={}
                for r in group:
                    key=r['label_member']
                    if key not in cache:
                        raw=z.read(key)
                        if hashlib.sha256(raw).hexdigest()!=r['label_sha256']:raise ValueError('Label checksum mismatch')
                        cache[key]=json.loads(raw)
                    u=cache[key]['utterance'][r['utterance_index']]
                    a,b=u.get('dialect_form'),u.get('standard_form')
                    if not isinstance(a,str) or not isinstance(b,str) or not a.strip() or not b.strip():continue
                    pairs.append(dict(sample_id=r['sample_id'],region=r['region'],audio_path=r['audio_path'],dialect=a,standard=b))
        write(OUT/f'{split}.json',dict(manifest_sha256=lock,rows=pairs))
    train=json.loads((OUT/'train.json').read_text('utf-8'))['rows']
    candidates=defaultdict(Counter)
    for r in train:
        a,b=normalize(r['dialect']).split(),normalize(r['standard']).split()
        for tag,i,j,k,l in SequenceMatcher(None,a,b,autojunk=False).get_opcodes():
            if tag!='replace' or not 1<=j-i<=3 or not 1<=l-k<=3:continue
            source,target=tuple(a[i:j]),tuple(b[k:l])
            # Exclude anonymization markers, unintelligible labels and number edits.
            if any(any(c.isdigit() for c in t) or not all('\uac00'<=c<='\ud7a3' for c in t) for t in source+target):continue
            candidates[source][target]+=1
    occurrences=Counter()
    for r in train:
        tokens=normalize(r['dialect']).split()
        for i in range(len(tokens)):
            for n in (1,2,3):
                key=tuple(tokens[i:i+n])
                if len(key)==n and key in candidates:occurrences[key]+=1
    rules=[]
    for source,targets in sorted(candidates.items()):
        target,count=targets.most_common(1)[0]
        rules.append(dict(source=list(source),target=list(target),support=count,occurrences=occurrences[source],confidence=count/occurrences[source]))
    write(OUT/'rules.json',rules)

def edit(text,rules,support=2,confidence=.9):
    table={tuple(r['source']):r['target'] for r in rules if r['support']>=support and r['confidence']>=confidence}
    # Preserve punctuation/spacing outside replaced spans.
    import re
    spans=list(re.finditer(r'[^\W_]+',text,re.UNICODE))
    result=[];pos=0;i=0
    while i<len(spans):
        match=None
        for n in (3,2,1):
            if i+n>len(spans):continue
            if any(text[spans[j].end():spans[j+1].start()].strip() for j in range(i,i+n-1)):continue
            key=tuple(normalize(s.group()) for s in spans[i:i+n])
            if key in table:match=(n,table[key]);break
        if match:
            n,target=match;end=spans[i+n-1].end()
            result.extend((text[pos:spans[i].start()],' '.join(target)));pos=end;i+=n
        else:i+=1
    return ''.join(result)+text[pos:]

def evaluate(rows,hypotheses):
    result={}
    for group in ('all','changed','unchanged'):
        indices=[i for i,r in enumerate(rows) if group=='all' or (normalize(r['dialect'])!=normalize(r['standard']))==(group=='changed')]
        if not indices:continue
        refs=[rows[i]['standard'] for i in indices];hyps=[hypotheses[i] for i in indices]
        result[group]=dict(count=len(indices),**error_rates(refs,hyps),exact_match=sum(normalize(a)==normalize(b) for a,b in zip(refs,hyps))/len(indices))
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['prepare','evaluate','asr']);args=p.parse_args()
    if args.command=='prepare':prepare();return
    rows=json.loads((OUT/'validation.json').read_text('utf-8'))['rows']
    rules=json.loads((OUT/'rules.json').read_text('utf-8'))
    if args.command=='asr':
        import torch
        from pipeline import Pipeline
        torch.manual_seed(42)
        engine=Pipeline();cache=OUT/'asr_validation.jsonl'
        done={}
        if cache.exists():
            for line in cache.read_text('utf-8').splitlines():
                r=json.loads(line);done[r['sample_id']]=r
        engine.run(rows[0]['audio_path'])
        with cache.open('a',encoding='utf-8') as stream:
            for i,r in enumerate(rows):
                if r['sample_id'] in done:continue
                result=engine.run(r['audio_path']);result['sample_id']=r['sample_id']
                stream.write(json.dumps(result,ensure_ascii=False)+'\n');stream.flush()
                if (i+1)%50==0:print(f'Validation ASR {i+1}/{len(rows)}',flush=True)
        return
    report={'selection_split':'validation','test_used':False,'identity':evaluate(rows,[r['dialect'] for r in rows]),'candidates':[]}
    old_rules=json.loads((ROOT/'data/standardization_v1/train_lexicon.json').read_text('utf-8'))
    report['v1_text']=evaluate(rows,[apply_lexicon(r['dialect'],old_rules) for r in rows])
    # Small declared grid, selected on validation only. Scores are development scores.
    for support in (2,3,5):
        for confidence in (.8,.9,1.0):
            hypotheses=[edit(r['dialect'],rules,support,confidence) for r in rows]
            report['candidates'].append(dict(support=support,confidence=confidence,metrics=evaluate(rows,hypotheses)))
    best=min(report['candidates'],key=lambda x:x['metrics']['all']['cer'])
    report['selected']=best
    cache=OUT/'asr_validation.jsonl'
    if cache.exists():
        predictions={r['sample_id']:r for r in map(json.loads,cache.read_text('utf-8').splitlines())}
        if set(predictions)!=set(r['sample_id'] for r in rows):raise ValueError('Incomplete ASR validation')
        texts=[predictions[r['sample_id']]['transcript'] for r in rows]
        edited=[edit(t,rules,best['support'],best['confidence']) for t in texts]
        report['audio_identity']=evaluate(rows,texts)
        report['audio_v1']=evaluate(rows,[apply_lexicon(t,old_rules) for t in texts])
        report['audio_edited']=evaluate(rows,edited)
        report['asr_vs_dialect']=error_rates([r['dialect'] for r in rows],texts)
        report['audio_per_region']={region:evaluate([r for r in rows if r['region']==region],
            [t for r,t in zip(rows,edited) if r['region']==region]) for region in sorted({r['region'] for r in rows})}
        write(OUT/'predictions.json',[dict(sample_id=r['sample_id'],transcript=t,standard_candidate=e) for r,t,e in zip(rows,texts,edited)])
    write(OUT/'evaluation.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='candidates'},ensure_ascii=True),flush=True)

if __name__=='__main__':main()
