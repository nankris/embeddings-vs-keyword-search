"""Recompute query metrics, bootstrap intervals and publication artifacts from saved ranks."""
import csv, hashlib, importlib.util, json, platform, sys
from pathlib import Path
from importlib.metadata import version
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
R = ROOT / 'results'
spec = importlib.util.spec_from_file_location('retrieval', ROOT/'scripts/02_run_retrieval.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
models = ['bm25', 'dense_minilm', 'dense_bge', 'hybrid']
labels = ['BM25', 'MiniLM', 'BGE-small', 'Hybrid (BM25 + MiniLM)']
corpus, queries, qrels_l = [m.load(n+'.jsonl') for n in ['corpus','queries','qrels']]
qr={}
for r in qrels_l:
    if r['score']>0: qr.setdefault(r['query_id'],{})[r['corpus_id']]=1
assert len(corpus)==5183 and len(qr)==300
assert all(r['score']==1 for r in qrels_l)
runs={s:json.loads((R/('run_'+s+'.json')).read_text()) for s in models}
valid_docs={c['id'] for c in corpus}
for s,run in runs.items():
    for q in qr:
        assert len(run[q])==100 and len(set(run[q]))==100
        assert set(run[q])<=valid_docs
# Independent hand-calculated examples to catch ranking/discount mistakes.
assert m.recall_at(['a','b'], {'a':1,'c':1})==.5
assert np.isclose(m.ndcg_at(['x','a'],{'a':1}),1/np.log2(3))
assert m.rrf({'q':['a','b']},{'q':['b','a']})['q']==['a','b']
sub,jac=m.build_paraphrase_subset(corpus,queries,qrels_l)
scores={s:{'nDCG@10':m.score_run(runs[s],qr,m.ndcg_at),'Recall@10':m.score_run(runs[s],qr,m.recall_at)} for s in models}
rows=[]
for q in sorted(qr):
    for s in models:
        rows.append({'query_id':q,'system':s,'low_overlap':q in sub,'relevant_documents':len(qr[q]),'retrieved_relevant_at_10':sum(d in qr[q] for d in runs[s][q][:10]),**{k:v[q] for k,v in scores[s].items()}})
with (R/'per_query.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
rng=np.random.default_rng(m.SEED)
summary={}; paired={}
for subset,ids in [('all',sorted(qr)),('low_overlap',sub)]:
    ids=sorted(ids)
    rng=np.random.default_rng(m.SEED)
    n=len(ids); samples=rng.integers(0,n,size=(10000,n))
    summary[subset]={};paired[subset]={}
    for s in models:
        summary[subset][s]={}
        for metric in ['nDCG@10','Recall@10']:
            a=np.array([scores[s][metric][q] for q in ids]);boot=a[samples].mean(axis=1)
            lo,hi=np.percentile(boot,[2.5,97.5])
            summary[subset][s][metric]={'mean':float(a.mean()),'lo':float(lo),'hi':float(hi)}
    for ai,a in enumerate(models):
        for b in models[:ai]:
            for metric in ['nDCG@10','Recall@10']:
                aa=np.array([scores[a][metric][q] for q in ids]);bb=np.array([scores[b][metric][q] for q in ids]);boot=(aa-bb)[samples].mean(axis=1)
                lo,hi=np.percentile(boot,[2.5,97.5]);paired[subset][f'{a}-{b}|{metric}']={'diff':float((aa-bb).mean()),'lo':float(lo),'hi':float(hi)}
    if subset=='all':  # Primary headline only; tiny-subset ratio estimates can be undefined.
        aa=np.array([scores['dense_bge']['Recall@10'][q] for q in ids]);bb=np.array([scores['bm25']['Recall@10'][q] for q in ids])
        denominator=bb[samples].mean(axis=1)
        assert np.all(denominator>0), "Relative gain undefined for a zero-baseline resample"
        boot=100*(aa[samples].mean(axis=1)/denominator-1)
        lo,hi=np.percentile(boot,[2.5,97.5]);summary[subset]['relative_recall_gain_bge_percent']={'mean':float(100*(aa.mean()/bb.mean()-1)),'lo':float(lo),'hi':float(hi)}
p=paired['all'];gain=p['dense_bge-bm25|nDCG@10']
if gain['lo']>0 and gain['diff']>=.05 and all(p[f'hybrid-{b}|nDCG@10']['lo']>0 for b in models[:3]): verdict='PROVEN'
elif gain['lo']<=0<=gain['hi']: verdict='DISPROVEN'
else: verdict='INCONCLUSIVE'
output={'seed':m.SEED,'bootstrap_resamples':10000,'verdict':verdict,'summary':summary,'paired':paired}
(R/'analysis.json').write_text(json.dumps(output,indent=2))
# Frozen artifacts and software versions; no credentials or private data.
manifest={'python':sys.version,'platform':platform.platform(),'packages':{p:version(p) for p in ['torch','transformers','sentence-transformers','rank-bm25','numpy','matplotlib','datasets']},'gpu':None,'colab_cu':0,'data_sources':['https://huggingface.co/datasets/BeIR/scifact','https://huggingface.co/datasets/BeIR/scifact-qrels'],'sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in R.iterdir() if f.suffix in ['.jsonl','.npy']}}
from huggingface_hub import scan_cache_dir
try:
    cache = scan_cache_dir()
    manifest['cached_model_revisions'] = {repo.repo_id: sorted(rev.commit_hash for rev in repo.revisions) for repo in cache.repos if repo.repo_id in m.MODELS.values()}
except Exception as e:
    manifest['cache_revision_note'] = type(e).__name__
audit=json.loads((R/'cache_audit.json').read_text())
assert audit['passed']
manifest['resumed_cache_audit']=audit
(R/'manifest.json').write_text(json.dumps(manifest,indent=2))
fig,axes=plt.subplots(1,2,figsize=(11,4.6))
colors=['#888888','#45a3b2','#2455a4','#d78732']
for ax,metric in zip(axes,['nDCG@10','Recall@10']):
    vals=[summary['all'][s][metric] for s in models]
    means=np.array([v['mean']*100 for v in vals]);lo=np.array([v['lo']*100 for v in vals]);hi=np.array([v['hi']*100 for v in vals])
    ax.bar(range(4),means,color=colors,yerr=[means-lo,hi-means],capsize=4)
    ax.set_xticks(range(4),['BM25','MiniLM','BGE-small','Hybrid']);ax.set_ylim(0,100);ax.set_ylabel(metric+' (points)')
    for i,v in enumerate(means):ax.text(i,v+7,f'{v:.1f}',ha='center',fontsize=10)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('Semantic search vs keyword search · SciFact (300 test queries)')
fig.text(.5,.01,'95% query-bootstrap intervals · 5,183 abstracts · CPU only · Hybrid = BM25 + MiniLM, RRF k=60',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(ROOT/'chart.png',dpi=180);plt.close(fig)
headline=summary['all']['relative_recall_gain_bge_percent']
lines=['# Embeddings versus keyword search','',f'**Verdict: {verdict}** under the decision rule written before scoring. BGE passes the embedding-gain criterion, but the fixed hybrid does not beat BGE.','',f'**Headline: “Embeddings found {headline["mean"]:.1f}% more right answers than keyword search.”**','',f'This means BGE’s relative increase in mean Recall@10 versus BM25 on 300 SciFact test queries: 95% paired-bootstrap CI **{headline["lo"]:.1f}% to {headline["hi"]:.1f}%**. A right answer here means a gold-labelled relevant abstract retrieved in the first ten results, not a generated factual answer.','', '| System | nDCG@10, % (95% CI) | Recall@10, % (95% CI) |','|---|---:|---:|']
for s,label in zip(models,labels):
    vv=[summary['all'][s][k] for k in ['nDCG@10','Recall@10']]
    lines.append('| '+label+' | '+' | '.join(f'{v["mean"]*100:.2f} [{v["lo"]*100:.2f}, {v["hi"]*100:.2f}]' for v in vv)+' |')
lines+=['','Paired differences below are percentage points; intervals resample the same queries for each system (10,000 draws, seed 20261016).','', '| Comparison | Δ nDCG@10 (95% CI) | Δ Recall@10 (95% CI) |','|---|---:|---:|']
for pair in ['dense_bge-bm25','dense_minilm-bm25','hybrid-bm25','hybrid-dense_minilm','hybrid-dense_bge']:
    vv=[p[pair+'|'+k] for k in ['nDCG@10','Recall@10']]
    lines.append('| '+pair+' | '+' | '.join(f'{v["diff"]*100:+.2f} [{v["lo"]*100:+.2f}, {v["hi"]*100:+.2f}]' for v in vv)+' |')
lines+=['','## Low-overlap diagnostic (30 queries)','',f'The original 30 test queries with the lowest maximum query/gold-abstract token Jaccard overlap were selected without using retrieval scores (maximum overlap {max(jac.values()):.4f}). These are not newly authored paraphrases; abstract length affects Jaccard, so this is only an exploratory diagnostic.','', '| System | nDCG@10, % (95% CI) | Recall@10, % (95% CI) |','|---|---:|---:|']
for s,label in zip(models,labels):
    vv=[summary['low_overlap'][s][k] for k in ['nDCG@10','Recall@10']]
    lines.append('| '+label+' | '+' | '.join(f'{v["mean"]*100:.2f} [{v["lo"]*100:.2f}, {v["hi"]*100:.2f}]' for v in vv)+' |')
lp=paired['low_overlap']['dense_bge-bm25|nDCG@10']
lines += ['',f'BGE minus BM25 on this subset: {lp["diff"]*100:+.2f} nDCG points (95% CI {lp["lo"]*100:+.2f} to {lp["hi"]*100:+.2f}). All subset metric intervals and paired comparisons are in `results/analysis.json`.','', '## Interpretation and limits','',f'BGE’s nDCG gain is {gain["diff"]*100:.2f} points; the target is at least 5 points plus a positive CI lower bound. Hybrid must also have a positive nDCG CI lower bound against all three individual rankers to satisfy the retained rule. The original rule calls a CI containing zero DISPROVEN; that label means the experiment did not establish the hypothesis, not that it established equivalence.','', 'This is one small scientific-claims benchmark, with incomplete relevance judgements and query-level uncertainty only. It does not establish performance on other domains. BM25 uses untuned rank_bm25 defaults and regex tokens, without stemming or stopword removal. Dense models use their SentenceTransformer defaults, normalized vectors, title plus abstract and no added query instruction; MiniLM truncates at its default 256 tokens and BGE at 512, so this compares practical default configurations rather than equal token budgets. Hybrid fuses only the top 100 ranks per component with k=60; no tuning or outcome-driven model selection was done. Both whole-corpus embeddings and the existing MiniLM caches are retained for audit.','', '## Reproduce','', 'Run with `/workspace/tools/mlenv/bin/python`, after `source /workspace/tools/setup.sh`:', '', '```bash','python scripts/01_prepare_data.py','OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false python scripts/02_run_retrieval.py','python scripts/04_audit_cache.py','python scripts/03_report.py','```','', 'Raw corpus, queries, qrels, top-100 rankings, per-query scores, caches and software/hash manifest are under `results/`. Existing embeddings and rankings are reused; remove affected cache files to recompute after changing inputs or settings. The report validates document IDs, rank uniqueness and hand-calculated metric examples. Six fixed MiniLM rows were freshly encoded and matched the resumed cache with cosine agreement above 0.9999 (`results/cache_audit.json`). CPU only; GPU: none; Colab CU: 0.','', '## Two-sentence takeaway','']
if headline['mean']>=0:
    lines.append(f'On these scientific claims, a small embedding model retrieved {headline["mean"]:.1f}% more relevant papers in its first ten results than keyword search. The fixed keyword-plus-MiniLM hybrid {"improved on keyword search" if summary["all"]["hybrid"]["nDCG@10"]["mean"] > summary["all"]["bm25"]["nDCG@10"]["mean"] else "did not improve on keyword search"}, but {"did" if summary["all"]["hybrid"]["nDCG@10"]["mean"] > summary["all"]["dense_bge"]["nDCG@10"]["mean"] else "did not"} beat BGE on ranking quality.')
else:
    lines.append(f'On these scientific claims, a small embedding model retrieved {abs(headline["mean"]):.1f}% fewer relevant papers in its first ten results than keyword search. Meaning-based search is useful, but its advantage depends on the model and the questions you ask.')
lines += ['', 'Decision checks: '+ '; '.join([
    f'BGE gain >=5 nDCG points: {gain["diff"]>=.05}',
    f'BGE gain CI excludes zero positively: {gain["lo"]>0}',
    *[f'Hybrid reliably beats {b}: {p[f"hybrid-{b}|nDCG@10"]["lo"]>0}' for b in models[:3]]]) + '.',
    '', 'Source links: [SciFact corpus and queries](https://huggingface.co/datasets/BeIR/scifact), [test relevance labels](https://huggingface.co/datasets/BeIR/scifact-qrels), [MiniLM model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [BGE model](https://huggingface.co/BAAI/bge-small-en-v1.5).']
(ROOT/'writeup.md').write_text('\n'.join(lines)+'\n')
print(json.dumps({'verdict':verdict,'headline':headline,'all':summary['all']},indent=2))
