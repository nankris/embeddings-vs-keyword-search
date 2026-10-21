"""Build the publication post from frozen protocol and measured statistics."""
import json,re
from pathlib import Path
R=Path('results');a=json.loads((R/'analysis.json').read_text());sel=json.loads((R/'pilot_selection.json').read_text());p=a['primary'];gain=p['mean']*100;lo,hi=[x*100 for x in p['ci95']]
verdict=a['verdict'];names={'bm25':'BM25','bge':'BGE-small','qwen':'Qwen 0.6B'}
if gain<0:title=f'Tuning hybrid search on SciFact still cost {abs(gain):.2f} points on fresh benchmarks'
elif verdict=='PROVEN':title='Pilot-tuned hybrid search cleared a two-point gain on fresh benchmarks'
elif verdict=='DISPROVEN':title='Pilot-tuned hybrid search improved too little to clear a two-point bar'
else:title='Pilot-tuned hybrid search left a two-point gain unresolved'
rule_interpretation={'PROVEN':'The entire interval cleared the practical margin. The pre-registered hypothesis was PROVEN on this benchmark mix.', 'DISPROVEN':'Even the upper endpoint was below the practical margin. The pre-registered hypothesis was DISPROVEN on this benchmark mix.', 'INCONCLUSIVE':'The interval crossed the practical margin. The pre-registered hypothesis remained INCONCLUSIVE on this benchmark mix.'}[verdict]
paragraphs=[]
for d,label,domain in [('fiqa','FiQA','financial questions'),('nfcorpus','NFCorpus','medical information needs'),('arguana','ArguAna','argument matching')]:
 z=a['datasets'][d];s=z['systems'];b=z['best_single'];v=z['hybrid_qwen_minus_best'];cl,ch=[x*100 for x in v['ci95']]
 paragraphs.append(f"**{label}**, covering {domain}, had {names[b]} as its strongest standalone retriever at {s[b]['ndcg10']['mean']*100:.2f}% nDCG@10. The Qwen hybrid scored {s['hybrid_qwen']['ndcg10']['mean']*100:.2f}%, a {v['mean']*100:+.2f}-point change, with a paired 95% interval of {cl:+.2f} to {ch:+.2f} points.")
results='\n\n'.join(paragraphs)
recall='; '.join(f"{label}: {a['datasets'][d]['systems']['qwen']['recall100']['mean']*100:.1f}% to {a['datasets'][d]['systems']['hybrid_qwen']['recall100']['mean']*100:.1f}%" for d,label in [('fiqa','FiQA'),('nfcorpus','NFCorpus'),('arguana','ArguAna')])
t=json.loads((R/'timing.json').read_text());stop=json.loads((R/'stopped.json').read_text())
text=f'''# {title}

*A SciFact pilot, two embedding models, and 2,377 fresh BEIR queries.*

You have a working vector search service. Someone suggests adding BM25 and merging the results. The pitch is appealing: exact keywords catch what embeddings miss, while embeddings handle different ways of saying the same thing.

The classic [Reciprocal Rank Fusion paper](https://doi.org/10.1145/1571941.1572114) says combining rankings “consistently yields better results than any individual system.” Its experiments support that claim in their settings. The engineering question is whether a useful improvement survives when your mixture moves to another domain.

**My headline result: {gain:+.2f} percentage points of nDCG@10 versus the strongest standalone retriever**, averaged equally across three fresh datasets. The paired 95% confidence interval was {lo:+.2f} to {hi:+.2f} points. {rule_interpretation}

That verdict concerns a two-point improvement, not every possible use of hybrid search.

## What I tested

This follows experiment 120. On SciFact, BGE-small beat basic BM25, but a fixed BM25-plus-MiniLM hybrid failed to beat BGE. This time I used better ingredients and tuned fusion before testing it elsewhere.

The five systems were BM25, [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5), [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), and a BM25 hybrid for each embedding model. Qwen's hybrid was the primary hypothesis; BGE's was secondary.

Dense retrieval used normalized embeddings and exact cosine search across full corpora. Both models had a 512-token input cap and retrieval instructions on queries. BGE ran in float32 and Qwen in float16 on an NVIDIA L4. BM25 used rank_bm25 defaults with lowercased word tokens, computed locally on CPU.

SciFact's 300 queries were development data. Confirmation used 648 FiQA queries against 57,638 documents, 323 NFCorpus queries against 3,633 documents, and 1,406 ArguAna queries against 8,674 documents. These public Hugging Face BEIR datasets were fresh to this experiment series, not necessarily absent from model training. ArguAna self matches were excluded for every system.

![Pilot and confirmation pipeline](figures/diagram.png)

Settings could change on SciFact, but they could not change after seeing the three confirmation datasets. That boundary is the main safeguard against turning benchmark feedback into apparent generalization.

## What tuning actually chose

Reciprocal Rank Fusion combines positions in ranked lists rather than raw BM25 and cosine scores. Its smoothing constant, k, controls how sharply it rewards top positions. A dense weight controls the embedding retriever's share of the combined score.

For each model, I tested four k values and five weights on SciFact, selecting the highest nDCG@10. Fusion considered the top 1,000 results from each component. Both selected k=10. BGE selected an 80% dense weight; Qwen selected 50%.

The tuned Qwen hybrid scored 72.74% on the pilot. The strongest standalone system, BGE, scored 71.27%. The development gain was only 1.47 points, already below the practical threshold. Each model had twenty configurations, giving tuning opportunities to capitalize on noise.

I kept the original hypothesis anyway: the Qwen hybrid would improve by at least two points on fresh data. The margin was an engineering hurdle for maintaining two retrieval paths, not a magnitude established by the paper.

## The fresh results

![Ranking scores on three fresh datasets](figures/scores.png)

This chart compares all five systems on nDCG@10, which rewards relevant documents near the top of the first ten results. Error bars represent query-level uncertainty, not uncertainty across future domains.

{results}

The comparator is deliberately demanding: the best of BM25, BGE and Qwen within each dataset, chosen after scoring. It tests the alternative I would want hybrid to beat, rather than comparing only with a weak ingredient.

![Hybrid gain against the strongest standalone system](figures/gains.png)

The dashed line marks the pre-registered two-point margin. The pooled row gives each dataset equal weight, so ArguAna's larger query count cannot dominate the headline.

I used 10,000 paired bootstrap draws, resampling queries within datasets. The primary interval repeats baseline selection in every draw. Dataset intervals hold their observed comparator fixed and are descriptive. PROVEN required the lower endpoint to reach two points; an upper endpoint below two meant DISPROVEN; crossing two meant INCONCLUSIVE.

![Coverage of relevant documents in the first hundred results](figures/recall.png)

Recall@100 asks whether relevant candidates were recovered; for Qwen versus its hybrid, it changed as follows: {recall}. Candidate coverage and top-ten ordering should be evaluated separately.

## What surprised me / what I got wrong

The losses were larger than I expected. On FiQA, fusion improved 105 queries relative to Qwen but hurt 281; on ArguAna, it improved 161 and hurt 684. These are exploratory counts, not another significance test.

The assumption under test was that a better fusion setting learned on scientific claims would deliver a useful gain elsewhere. The pilot gave reason for caution before confirmation: even its selected point estimate missed two points against BGE.

I also couldn't assume the larger embedding model would be the strongest baseline everywhere. That's why the test included both dense models and BM25. Calling an improvement over one ingredient a deployment win would have repeated the comparison mistake from experiment 120.

The committed verdict stays {verdict}. A favorable secondary comparison cannot replace the primary test after the results arrive.

## What I would do in production

1. **Establish the strongest feasible standalone baseline.** Include a dense-only fallback in the tuning grid. This grid required at least 20% lexical weight, so it could not choose to switch BM25 off. Compare under identical relevance rules.
2. **Keep development and release tests separate.** Tune on representative development queries, then evaluate untouched queries and domains before shipping the fusion settings.
3. **Track ranking and candidate coverage separately.** More candidates can help a later reranker, but this experiment didn't test one or measure generated-answer quality.
4. **Measure the complete operating cost.** Include lexical indexing, query encoding, both searches and merging. Batch encoding times here don't establish online p95 latency.

## Limitations

This was weighted RRF with two components, not a reproduction of the original paper's collections and system ensembles. BM25 had no stemming, stopword removal or parameter tuning. A stronger lexical implementation could change the result.

The common 512-token cap constrained Qwen's longer-context capability. Different tokenizers also preserve different spans, and the generic query instruction may fit argument matching poorly. Those choices remained fixed through confirmation.

Three datasets cannot represent every production domain. Five ArguAna gold document IDs were missing from the HF corpus. I retained those five queries, which scored zero for every system. Labels are incomplete, query resampling assumes independence, and model training may have included these benchmarks. Exact search also avoids the approximation and scaling choices a larger service would face.

Timing: estimated 100 minutes; actual {t['actual_minutes']:.1f} minutes, including the earlier allocation wait and interruption. NVIDIA L4 usage was approximately {stop['estimated_cu']:.3f} CU, estimated from session duration at 1.54 CU/hour. The detached worker saved checkpoints and the session was stopped after completion.

A fusion setting earns its place by surviving a fresh test. Its pilot score is only the invitation to run that test.

Code: <repo link placeholder>
'''
Path('blog.md').write_text(text)
print('Blog word count:',len(re.findall(r"\b[\w'-]+\b",text)))
