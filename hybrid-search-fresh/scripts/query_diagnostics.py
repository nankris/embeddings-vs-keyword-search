"""Post-hoc descriptive diagnostics, never used for tuning or the primary verdict."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
out={}
for ds in ['fiqa','nfcorpus','arguana']:
 p=Path('results')/ds;df=pd.read_csv(p/'metrics.csv',dtype={'qid':str});v=df.pivot(index='qid',columns='system',values='ndcg10');d=json.loads((p/'evaluation_inputs.json').read_text());q=dict(zip(d['qids'],d['queries']))
 delta=v.hybrid_qwen-v.qwen
 out[ds]={'exploratory':True,'queries_improved':int((delta>1e-12).sum()),'queries_hurt':int((delta<-1e-12).sum()),'queries_unchanged':int((abs(delta)<=1e-12).sum()),'gain_vs_qwen':float(delta.mean()),'largest_gain_examples':[{'qid':str(i),'query':q[str(i)],'ndcg_change':float(delta.loc[i])} for i in delta.nlargest(2).index],'largest_loss_examples':[{'qid':str(i),'query':q[str(i)],'ndcg_change':float(delta.loc[i])} for i in delta.nsmallest(2).index]}
Path('results/query_diagnostics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
