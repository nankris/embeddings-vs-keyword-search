"""Additional Recall@100 chart and paired secondary contrasts from saved query scores."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('results');a=json.loads((R/'analysis.json').read_text())
ds=['fiqa','nfcorpus','arguana'];systems=['bm25','bge','qwen','hybrid_bge','hybrid_qwen']
names=['BM25','BGE-small','Qwen 0.6B','Hybrid BGE','Hybrid Qwen']
plt.rcParams.update({'font.size':12,'savefig.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(11,5.5));x=np.arange(3);w=.15
for j,s in enumerate(systems):
 z=[a['datasets'][d]['systems'][s]['recall100'] for d in ds]
 y=np.array([t['mean'] for t in z])*100;c=np.array([t['ci95'] for t in z])*100
 ax.bar(x+(j-2)*w,y,w,label=names[j],color=['#8d99ae','#457b9d','#264653','#e9c46a','#e76f51'][j])
 ax.errorbar(x+(j-2)*w,y,yerr=[y-c[:,0],c[:,1]-y],fmt='none',capsize=2,color='#333333',lw=.8)
ax.set(xticks=x,xticklabels=['FiQA','NFCorpus','ArguAna'],ylabel='Recall@100 (%)',ylim=(0,105),title='Candidate coverage is a separate question from top-ten ranking')
ax.legend(ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.1));fig.tight_layout();fig.savefig('figures/recall.png',bbox_inches='tight');plt.close(fig)
rng=np.random.default_rng(12020261019);out={}
for d in ds:
 df=pd.read_csv(R/d/'metrics.csv',dtype={'qid':str});out[d]={}
 for metric in ['ndcg10','recall100']:
  v=df.pivot(index='qid',columns='system',values=metric);base=a['datasets'][d]['best_single']
  for h,b in [('hybrid_qwen','qwen'),('hybrid_bge','bge'),('hybrid_bge',base),('hybrid_qwen',base)]:
   delta=(v[h]-v[b]).values;boot=np.concatenate([delta[rng.integers(0,len(delta),(250,len(delta)))].mean(1) for _ in range(40)])
   out[d][h+' minus '+b+' '+metric]={'mean':float(delta.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist()}
(R/'secondary_contrasts.json').write_text(json.dumps(out,indent=2))
