"""Download pinned public model files with curl; no audio/text leaves this machine."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]
MODELS={'asr':'openai/whisper-small','rewrite':'Qwen/Qwen2.5-1.5B-Instruct',
        'asr_turbo':'openai/whisper-large-v3-turbo'}
REVISIONS={'asr':'973afd24965f72e36ca33b3055d56a652f456b4d',
           'rewrite':'989aa7980e4cf806f80c7fef2b1adb7bc71aa306',
           'asr_turbo':'41f01f3fe87f28c78e2fbf8b568835947dd65ed9'}
NAMES={'config.json','generation_config.json','preprocessor_config.json','tokenizer.json',
       'tokenizer_config.json','special_tokens_map.json','added_tokens.json','vocab.json',
       'merges.txt','normalizer.json','model.safetensors','model.safetensors.index.json'}

def fetch(url,path):
    subprocess.run(['curl.exe' if __import__('os').name=='nt' else 'curl','--fail','--location','--silent','--show-error',
                    '--connect-timeout','30','--max-time','1800',url,'-o',str(path)],check=True)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('kind',choices=MODELS)
    args=parser.parse_args()
    out=ROOT/'data/standardization_models'/args.kind
    out.mkdir(parents=True,exist_ok=True)
    info_path=out/'upstream.json'
    if not info_path.exists():
        temporary=out/'upstream.partial'
        fetch(f'https://huggingface.co/api/models/{MODELS[args.kind]}/revision/{REVISIONS[args.kind]}?blobs=true',temporary)
        json.loads(temporary.read_text('utf-8'))
        temporary.replace(info_path)
    info=json.loads(info_path.read_text('utf-8'))
    if info['sha']!=REVISIONS[args.kind]:raise ValueError('Model revision differs from the pinned baseline')
    entries=[e for e in info['siblings'] if e['rfilename'] in NAMES or
             (e['rfilename'].startswith('model-') and e['rfilename'].endswith('.safetensors'))]
    receipt=dict(model_id=MODELS[args.kind],revision=info['sha'],files={})
    for e in entries:
        name=e['rfilename'];target=out/name
        if not target.exists():
            print(f'Downloading {args.kind}/{name} ({e.get("size",0)} bytes)',flush=True)
            partial=out/(name+'.partial')
            fetch(f'https://huggingface.co/{MODELS[args.kind]}/resolve/{info["sha"]}/{name}',partial)
            actual=hashlib.file_digest(partial.open('rb'),'sha256').hexdigest()
            if 'lfs' in e and actual!=e['lfs']['sha256']:
                raise ValueError('Model file hash mismatch')
            if partial.stat().st_size!=e['size']:
                raise ValueError('Model file size mismatch')
            partial.replace(target)
        actual=hashlib.file_digest(target.open('rb'),'sha256').hexdigest()
        if 'lfs' in e and actual!=e['lfs']['sha256']:
            raise ValueError('Cached model hash mismatch')
        receipt['files'][name]=actual
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(f'{args.kind}: complete, revision={info["sha"]}',flush=True)

if __name__=='__main__':main()
