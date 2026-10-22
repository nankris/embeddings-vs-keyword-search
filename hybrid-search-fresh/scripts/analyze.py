"""Paired, stratified bootstrap. Comparator selection repeats in every primary draw."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('results');F=Path('figures');F.mkdir(exist_ok=True)
DS=['fiqa','nfcorpus','arguana'];SYS=['bm25','bge','qwen','hybrid_bge','hybrid_qwen'];NAMES=['BM25','BGE-small','Qwen 0.6B','Hybrid BGE','Hybrid Qwen']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','axes.titleweight':'bold','savefig.dpi':150})
def ci(v):return np.quantile(v,[.025,.975]).tolist()
def describe(v,b):return {'mean':float(v),'ci95':ci(b)}
rng=np.random.default_rng(12020261018);out={'bootstrap_draws':10000,'datasets':{}};boots={};means={};ns=[]
for ds in DS:
 df=pd.read_csv(R/ds/'metrics.csv',dtype={'qid':str});n=df.qid.nunique();ns.append(n)
 vals=np.stack([df.pivot(index='qid',columns='system',values=m)[SYS].values for m in ['ndcg10','recall100']],axis=2)
 assert vals.shape==(n,5,2) and np.isfinite(vals).all() and ((vals>=0)&(vals<=1+1e-9)).all()
 mean=vals.mean(0);boot=[]
 for a in range(0,10000,250):boot.append(vals[rng.integers(0,n,size=(min(250,10000-a),n))].mean(1))
 boot=np.concatenate(boot);boots[ds]=boot;means[ds]=mean
 best=int(np.argmax(mean[:3,0]));delta=mean[4,0]-mean[best,0]
 out['datasets'][ds]={'n':n,'systems':{s:{m:describe(mean[j,k],boot[:,j,k]) for k,m in enumerate(['ndcg10','recall100'])} for j,s in enumerate(SYS)},'best_single':SYS[best], 'hybrid_qwen_minus_best':describe(delta,boot[:,4,0]-boot[:,best,0]),'selection_repeated_ci95':ci(boot[:,4,0]-boot[:,:3,0].max(1))}
# Equal weight for each dataset; strongest single recomputed within each draw.
primary=np.mean([means[d][4,0]-means[d][:3,0].max() for d in DS])
pboot=np.mean([boots[d][:,4,0]-boots[d][:,:3,0].max(1) for d in DS],axis=0)
lo,hi=ci(pboot);verdict='PROVEN' if lo>=.02 else 'DISPROVEN' if hi<.02 else 'INCONCLUSIVE'
macro_mean=np.mean(list(means.values()),axis=0);macro_boot=np.mean(list(boots.values()),axis=0)
qmean=np.average(list(means.values()),axis=0,weights=ns);qboot=np.average(list(boots.values()),axis=0,weights=ns)
out.update({'primary':describe(primary,pboot),'verdict':verdict,'macro':{s:{m:describe(macro_mean[j,k],macro_boot[:,j,k]) for k,m in enumerate(['ndcg10','recall100'])} for j,s in enumerate(SYS)},'query_weighted_gain':describe(np.average([means[d][4,0]-means[d][:3,0].max() for d in DS],weights=ns),np.average([boots[d][:,4,0]-boots[d][:,:3,0].max(1) for d in DS],axis=0,weights=ns))})
(R/'analysis.json').write_text(json.dumps(out,indent=2))
rows=[]
for ds in DS:
 x=out['datasets'][ds];rows.append([{'fiqa':'FiQA','nfcorpus':'NFCorpus','arguana':'ArguAna'}[ds],str(x['n'])]+[f"{x['systems'][s]['ndcg10']['mean']*100:.2f}" for s in SYS]+[f"{x['hybrid_qwen_minus_best']['mean']*100:+.2f} [{x['hybrid_qwen_minus_best']['ci95'][0]*100:+.2f}, {x['hybrid_qwen_minus_best']['ci95'][1]*100:+.2f}]"])
rows.append(['Equal-weight',str(sum(ns))]+[f'{macro_mean[j,0]*100:.2f}' for j in range(5)]+[f'{primary*100:+.2f} [{lo*100:+.2f}, {hi*100:+.2f}]'])
headers=['Dataset','Queries','BM25','BGE','Qwen','H-BGE','H-Qwen','H-Qwen gain [95% CI]']
pd.DataFrame(rows,columns=headers).to_csv(R/'main_table.csv',index=False)
fig,ax=plt.subplots(figsize=(15,3.3));ax.axis('off');t=ax.table(cellText=rows,colLabels=headers,loc='center',cellLoc='center',colWidths=[.11,.065,.07,.07,.07,.07,.08,.30]);t.auto_set_font_size(False);t.set_fontsize(11);t.scale(1,2)
for (r,c),cell in t.get_celld().items():
 cell.set_edgecolor('white')
 if r==0:cell.set_facecolor('#243a53');cell.set_text_props(color='white',weight='bold')
 else:cell.set_facecolor('#eaf0f6' if r%2 else '#f6f8fa')
ax.set_title('Pilot-tuned fusion on three fresh BEIR datasets',pad=4,fontsize=17);fig.text(.05,.02,'nDCG@10 in %. Gain in percentage points versus strongest single system per dataset. Equal-weight row averages datasets.',fontsize=10);fig.savefig(F/'table.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(11,5.5));xx=np.arange(3);w=.15;colors=['#8d99ae','#457b9d','#264653','#e9c46a','#e76f51']
for j,s in enumerate(SYS):
 y=np.array([means[d][j,0]*100 for d in DS]);cis=np.array([out['datasets'][d]['systems'][s]['ndcg10']['ci95'] for d in DS])*100
 ax.bar(xx+(j-2)*w,y,w,label=NAMES[j],color=colors[j]);ax.errorbar(xx+(j-2)*w,y,yerr=[y-cis[:,0],cis[:,1]-y],fmt='none',ecolor='#333333',capsize=2,lw=.8)
ax.set_xticks(xx,['FiQA','NFCorpus','ArguAna']);ax.set_ylabel('nDCG@10 (%)');ax.set_title('One pilot setting, three different outcomes');ax.legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,-.1),frameon=False);ax.set_ylim(0,max(v[:,:][...,0].max() for v in means.values())*100+15);fig.tight_layout();fig.savefig(F/'scores.png',bbox_inches='tight');plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5));labels=['FiQA','NFCorpus','ArguAna','Equal-weight pooled'];ys=[];low=[];high=[]
for d in DS:
 z=out['datasets'][d]['hybrid_qwen_minus_best'];ys.append(z['mean']*100);low.append(z['ci95'][0]*100);high.append(z['ci95'][1]*100)
ys.append(primary*100);low.append(lo*100);high.append(hi*100)
ax.axvline(0,color='#888',lw=1);ax.axvline(2,color='#c44',ls='--',label='Preregistered +2 pp margin');ax.errorbar(ys,np.arange(4),xerr=[np.array(ys)-low,np.array(high)-ys],fmt='o',markersize=8,color='#264653',capsize=5);ax.set_yticks(range(4),labels);ax.invert_yaxis();ax.set_xlabel('Hybrid Qwen minus strongest single (nDCG percentage points)');ax.set_title('Does the hybrid clear a useful improvement?');ax.legend(loc='best',frameon=False);fig.text(.12,.01,'95% paired query bootstrap; pooled CI repeats strongest-system selection within each draw.',fontsize=10);fig.tight_layout(rect=(0,.04,1,1));fig.savefig(F/'gains.png',bbox_inches='tight');plt.close(fig)
print(json.dumps({'verdict':verdict,'primary':out['primary'],'dataset_gain':{d:out['datasets'][d]['hybrid_qwen_minus_best'] for d in DS}},indent=2))
