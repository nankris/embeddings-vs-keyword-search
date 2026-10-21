"""Recompute all published query metrics from saved ranking IDs and qrels."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd
from run import metrics,fuse
def independent_metrics(d,rr):
 values=[]
 for qi,row in zip(d['qids'],rr):
  relevant=d['qrels'][qi]
  top=[d['docids'][int(i)] for i in row]
  dcg=sum(relevant.get(doc,0)/math.log2(pos+2) for pos,doc in enumerate(top[:10]))
  idcg=sum(gain/math.log2(pos+2) for pos,gain in enumerate(sorted(relevant.values(),reverse=True)[:10]))
  recall=len(set(top[:100]).intersection(relevant))/len(relevant)
  values.append([dcg/idcg,recall])
 return np.asarray(values)
R=Path('results');sel=json.loads((R/'pilot_selection.json').read_text());summary={}
for ds in ['fiqa','nfcorpus','arguana']:
 p=R/ds;d=json.loads((p/'evaluation_inputs.json').read_text());df=pd.read_csv(p/'metrics.csv',dtype={'qid':str});rankings={}
 assert len(set(d['docids']))==len(d['docids']) and len(set(d['qids']))==len(d['qids'])
 docset=set(d['docids'])
 assert all(all(g>0 for g in rel.values()) for rel in d['qrels'].values())
 missing=[{'qid':q,'docid':doc} for q in d['qids'] for doc in d['qrels'][q] if doc not in docset]
 # Retain all preregistered queries and original qrels, including unreachable golds.
 assert len(missing)==(5 if ds=='arguana' else 0),missing
 for s in ['bm25','bge','qwen','hybrid_bge','hybrid_qwen']:
  rr=np.load(p/(s+'.npy'));rankings[s]=rr
  assert len(rr)==len(d['qids']) and rr.min()>=0 and rr.max()<len(d['docids'])
  assert all(len(set(row))==len(row) for row in rr)
  if ds=='arguana':assert all(q not in [d['docids'][i] for i in row] for q,row in zip(d['qids'],rr))
  got=independent_metrics(d,rr);expected=df[df.system==s].set_index('qid').loc[d['qids'],['ndcg10','recall100']].values
  assert np.allclose(got,expected,atol=1e-12)
 for s in ['bge','qwen']:
  assert np.array_equal(fuse(rankings['bm25'],rankings[s],sel[s]['k'],sel[s]['w']),rankings['hybrid_'+s])
 summary[ds]={'nqueries':len(d['qids']),'qrels_missing_from_corpus':missing,'systems_checked':5,'rankings_unique':True,'metrics_recomputed_independent_python':True,'fusion_recomputed':True}
grid=json.loads((R/'pilot_grid.json').read_text())
for s in ['bge','qwen']:assert max((g for g in grid if g['dense']==s),key=lambda g:g['ndcg10'])==sel[s]
(R/'audit.json').write_text(json.dumps(summary,indent=2));print('PASS: all ranking bounds, uniqueness, self exclusion, metrics, fusion and pilot selection')
