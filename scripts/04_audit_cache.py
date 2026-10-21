"""Audit resumed MiniLM embeddings against fresh encodings of fixed sample rows."""
import json, os
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np
import torch
torch.set_num_threads(2)
from sentence_transformers import SentenceTransformer
root=Path(__file__).resolve().parents[1];r=root/'results'
m=SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2',revision='1110a243fdf4706b3f48f1d95db1a4f5529b4d41',device='cpu')
result={'model':'sentence-transformers/all-MiniLM-L6-v2','max_seq_length':m.max_seq_length,'samples':{}}
for kind,indices in [('docs',[0,2500,5182]),('queries',[0,500,1108])]:
    items=[json.loads(x) for x in (r/('corpus.jsonl' if kind=='docs' else 'queries.jsonl')).read_text().splitlines()]
    texts=[(items[i]['title']+' '+items[i]['text']).strip() if kind=='docs' else items[i]['text'] for i in indices]
    new=m.encode(texts,normalize_embeddings=True,convert_to_numpy=True)
    old=np.load(r/f'emb_dense_minilm_{kind}_aac97e68.npy')[indices]
    sims=(new*old).sum(axis=1)
    result['samples'][kind]={'indices':indices,'cosine_agreement':sims.tolist()}
    assert sims.min()>.9999, 'Cache mismatch: invalidate and recompute MiniLM caches and rankings'
result['passed']=True
(r/'cache_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
