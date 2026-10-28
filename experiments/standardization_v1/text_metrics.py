import re
import unicodedata

def normalize(text):
    text=unicodedata.normalize('NFC',text)
    return ' '.join(''.join(c if c.isalnum() or c.isspace() else ' ' for c in text).split())

def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def error_rates(references,hypotheses):
    if not references or len(references)!=len(hypotheses):raise ValueError('Need equal nonempty lists')
    ce=we=nc=nw=0
    for ref,hyp in zip(references,hypotheses):
        ref,hyp=normalize(ref),normalize(hyp)
        a,b=ref.replace(' ',''),hyp.replace(' ','')
        ce+=distance(a,b);nc+=len(a)
        we+=distance(ref.split(),hyp.split());nw+=len(ref.split())
    return dict(cer=ce/nc if nc else None,wer=we/nw if nw else None,
                character_errors=ce,reference_characters=nc,word_errors=we,reference_words=nw)

def apply_lexicon(text,lexicon):
    if not lexicon:return text
    keys=sorted(lexicon,key=lambda s:(-len(s),s))
    pattern=r'(?<!\w)('+ '|'.join(re.escape(k) for k in keys)+r')(?!\w)'
    return re.sub(pattern,lambda m:lexicon[m.group(0)],text)
