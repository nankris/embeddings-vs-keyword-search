"""Detached Colab worker. Resume chunks and completed rankings; never retune on tests."""
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ.setdefault('HF_HOME',os.path.join(os.getcwd(),'hf_cache'))
os.environ['OMP_NUM_THREADS']='2'
import time,json,re,gc,hashlib,subprocess,sys,platform
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from huggingface_hub import HfApi,hf_hub_download
from rank_bm25 import BM25Okapi
R=Path('results');R.mkdir(exist_ok=True)
START=time.time()

def save(path,obj):
 p=Path(path); tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(p)
def save_array(path, array):
 p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('wb') as f:np.save(f,array)
 tmp.replace(p)
def save_chunk(path,emb,tokens):
 p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('wb') as f:np.savez(f,emb=emb,tokens=tokens)
 tmp.replace(p)
def log(s):print(datetime.now(timezone.utc).isoformat(),s,flush=True)
def data(ds):
 p=R/ds;p.mkdir(exist_ok=True)
 if (p/'data.json').exists():return json.loads((p/'data.json').read_text())
 api=HfApi(); repo='BeIR/'+ds
 pinned=json.loads((R/'data_manifest.json').read_text()).get(ds,{}) if (R/'data_manifest.json').exists() else {}
 rev=pinned.get('revision') or api.dataset_info(repo).sha;qrev=pinned.get('qrels_revision') or api.dataset_info(repo+'-qrels').sha
 def part(sub):
  files=[f for f in api.list_repo_files(repo,repo_type='dataset',revision=rev) if f.startswith(sub+'/') and f.endswith('.parquet')]
  return pd.concat([pd.read_parquet(hf_hub_download(repo,f,repo_type='dataset',revision=rev)) for f in sorted(files)],ignore_index=True)
 c=part('corpus').fillna('');q=part('queries').fillna('')
 rel=pd.read_csv(hf_hub_download(repo+'-qrels','test.tsv',repo_type='dataset',revision=qrev),sep='\t',dtype={'query-id':str,'corpus-id':str})
 rel=rel[rel.score>0]; ids=set(rel['query-id'])
 q=q[q['_id'].astype(str).isin(ids)].sort_values('_id')
 obj={'docids':c['_id'].astype(str).tolist(),'texts':[(str(x.get('title',''))+' '+str(x['text'])).strip() for x in c.to_dict('records')], 'qids':q['_id'].astype(str).tolist(),'queries':q['text'].tolist(),'qrels':{str(k):dict(zip(g['corpus-id'].astype(str),g['score'].astype(float))) for k,g in rel.groupby('query-id')},'revision':rev,'qrels_revision':qrev}
 assert len(obj['qids'])==len(ids)
 save(p/'data.json',obj);log(f'DATA {ds}: {len(c)} documents, {len(q)} queries');return obj

def rank(scores,k):
 # Stable complete sort makes ties deterministic even at cutoff.
 return np.argsort(-scores,kind='stable',axis=1)[:,:k].astype(np.int32)
def lexical(ds,d):
 p=R/ds;n=len(d['docids']);k=min(1000,n);docindex={v:i for i,v in enumerate(d['docids'])}
 def exclude(s,offset=0):
  for j,q in enumerate(d['qids'][offset:offset+len(s)]):
   if ds=='arguana' and q in docindex:s[j,docindex[q]]=-np.inf
 if not (p/'bm25.npy').exists():
  t=time.time();tok=lambda x:re.findall(r'\w+',x.lower())
  bm=BM25Okapi([tok(x) for x in d['texts']]);out=[];build_seconds=time.time()-t;tquery=time.time()
  for qi,q in enumerate(d['queries']):
   scores=np.asarray(bm.get_scores(tok(q)))[None,:];exclude(scores,qi);out.append(rank(scores,k)[0])
  save_array(p/'bm25.npy',np.stack(out));save(p/'bm25_cost.json',{'seconds':time.time()-t,'build_seconds':build_seconds,'query_seconds':time.time()-tquery,'nqueries':len(d['qids'])});log(f'{ds} BM25 seconds={time.time()-t:.2f}')

def single(ds,d):
 import torch
 from sentence_transformers import SentenceTransformer
 torch.set_num_threads(2)
 p=R/ds; n=len(d['docids']);k=min(1000,n);docindex={v:i for i,v in enumerate(d['docids'])}
 def exclude(s,offset=0):
  for j,q in enumerate(d['qids'][offset:offset+len(s)]):
   if ds=='arguana' and q in docindex:s[j,docindex[q]]=-np.inf
 lexical(ds,d)
 for short,name in [('bge','BAAI/bge-small-en-v1.5'),('qwen','Qwen/Qwen3-Embedding-0.6B')]:
  if (p/(short+'.npy')).exists():continue
  t=time.time();mp=R/'model_revisions.json';revs=json.loads(mp.read_text()) if mp.exists() else {}
  if name not in revs:revs[name]=HfApi().model_info(name).sha;save(mp,revs)
  rev=revs[name]
  model=SentenceTransformer(name,revision=rev,device='cuda',model_kwargs={'torch_dtype':torch.float16 if short=='qwen' else torch.float32})
  model.max_seq_length=512
  prefix='Represent this sentence for searching relevant passages: ' if short=='bge' else 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: '
  total_tokens=0;times={};token_kinds={}
  for kind,texts in [('docs',d['texts']),('queries',[prefix+x for x in d['queries']])]:
   cache=p/(short+'_'+kind);cache.mkdir(exist_ok=True);parts=[];tokens=0;tenc=time.time()
   for a in range(0,len(texts),512):
    cp=cache/f'{a:06d}.npz'
    if cp.exists():
     z=np.load(cp);parts.append(z['emb']);tokens+=int(z['tokens']);continue
    chunk=texts[a:a+512]
    tt=model.tokenize(chunk);ct=int(tt['attention_mask'].sum());del tt
    with torch.inference_mode():emb=model.encode(chunk,batch_size=32 if short=='qwen' else 64,normalize_embeddings=True,show_progress_bar=False,convert_to_numpy=True)
    emb=emb.astype(np.float32);assert np.isfinite(emb).all() and np.allclose(np.linalg.norm(emb,axis=1),1,atol=.003);save_chunk(cp,emb,ct);parts.append(emb);tokens+=ct
    log(f'ENCODE {ds} {short} {kind} {min(a+512,len(texts))}/{len(texts)} tokens={tokens} seconds={time.time()-tenc:.1f}')
   vec=np.concatenate(parts);times[kind]=time.time()-tenc;total_tokens+=tokens;token_kinds[kind]=tokens
   save_array(p/(short+'_'+kind+'_audit8.npy'),vec[:8])
   if kind=='docs':docvec=vec
   else:qvec=vec
  rankings=[];ts=time.time()
  dv=torch.from_numpy(docvec).cuda()
  for a in range(0,len(qvec),64):
   scores=(torch.from_numpy(qvec[a:a+64]).cuda()@dv.T).cpu().numpy();exclude(scores,a);rankings.append(rank(scores,k))
  save_array(p/(short+'.npy'),np.concatenate(rankings));cost={'seconds':time.time()-t,'encode_seconds':times,'search_seconds':time.time()-ts,'input_tokens_truncated_nonpadding':total_tokens,'tokens_by_kind':token_kinds,'output_tokens':0,'model':name,'revision':rev,'max_seq_length':512,'resumed_chunks_included_in_token_total':True}
  save(p/(short+'_cost.json'),cost);log(f'DONE {ds} {short}: {cost}')
  del model,dv,docvec,qvec;gc.collect();torch.cuda.empty_cache()
 return {s:np.load(p/(s+'.npy')) for s in ['bm25','bge','qwen']}

def fuse(bm,dense,k,w):
 out=[]
 for br,dr in zip(bm,dense):
  scores={int(doc):(1-w)/(k+i+1) for i,doc in enumerate(br)}
  for i,doc in enumerate(dr):scores[int(doc)]=scores.get(int(doc),0)+w/(k+i+1)
  out.append([doc for doc,s in sorted(scores.items(),key=lambda x:(-x[1],x[0]))[:100]])
 return np.array(out,dtype=np.int32)
def metrics(d,ranking):
 vals=[]
 for qi,rr in zip(d['qids'],ranking):
  rel=d['qrels'][qi];g=np.array([rel.get(d['docids'][i],0) for i in rr]);ideal=np.sort(list(rel.values()))[::-1][:10];idcg=(ideal/np.log2(np.arange(len(ideal))+2)).sum()
  vals.append([float((g[:10]/np.log2(np.arange(min(10,len(g)))+2)).sum()/idcg),float((g[:100]>0).sum()/len(rel))])
 return np.array(vals)

def main():
 log(subprocess.check_output(['nvidia-smi'],text=True));save(R/'environment.json',{'python':sys.version,'platform':platform.platform(),'packages':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv'],text=True)})
 if not (R/'pilot_selection.json').exists():
  d=data('scifact');r=single('scifact',d);grid=[];sel={}
  for s in ['bge','qwen']:
   for k in [10,30,60,100]:
    for w in [.20,.35,.50,.65,.80]:grid.append({'dense':s,'k':k,'w':w,'ndcg10':float(metrics(d,fuse(r['bm25'],r[s],k,w))[:,0].mean())})
   sel[s]=max([g for g in grid if g['dense']==s],key=lambda g:g['ndcg10'])
  save(R/'pilot_grid.json',grid);save(R/'pilot_selection.json',sel);log(f'PILOT FROZEN {sel}')
 sel=json.loads((R/'pilot_selection.json').read_text())
 for ds in ['fiqa','nfcorpus','arguana']:
  p=R/ds
  if (p/'metrics.csv').exists():log(f'RESUME completed {ds}');continue
  d=data(ds);r=single(ds,d)
  for s in ['bge','qwen']:
   r['hybrid_'+s]=fuse(r['bm25'],r[s],sel[s]['k'],sel[s]['w']);save_array(p/('hybrid_'+s+'.npy'),r['hybrid_'+s])
  rows=[]
  for s,ranking in r.items():
   m=metrics(d,ranking)
   for q,v in zip(d['qids'],m):rows.append({'dataset':ds,'qid':q,'system':s,'ndcg10':v[0],'recall100':v[1]})
  pd.DataFrame(rows).to_csv(p/'metrics.csv.tmp',index=False);(p/'metrics.csv.tmp').replace(p/'metrics.csv');log(f'SCORED {ds}')
 save(R/'worker_done.json',{'seconds':time.time()-START,'utc':datetime.now(timezone.utc).isoformat()});log('WORKER COMPLETE')
if __name__=='__main__':
 try:main()
 except Exception:
  import traceback
  traceback.print_exc();save(R/'worker_error.json',{'utc':datetime.now(timezone.utc).isoformat(),'error':traceback.format_exc()});raise
