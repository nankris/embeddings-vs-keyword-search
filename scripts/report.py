"""Render concise measured writeup after analysis; preserve original preregistration."""
import json
from pathlib import Path
R=Path('results');a=json.loads((R/'analysis.json').read_text());sel=json.loads((R/'pilot_selection.json').read_text());tim=json.loads((R/'timing.json').read_text());stop=json.loads((R/'stopped.json').read_text())
SYS=['bm25','bge','qwen','hybrid_bge','hybrid_qwen'];DS=['fiqa','nfcorpus','arguana']
def fmt(z,sign=False):
 spec='+.2f' if sign else '.2f'
 return f"{format(z['mean']*100,spec)} [{format(z['ci95'][0]*100,spec)}, {format(z['ci95'][1]*100,spec)}]"
headline=f"{a['primary']['mean']*100:+.2f} nDCG@10 percentage points versus the strongest single retriever (equal dataset weights)"
timing=f"Timing: estimated 100 min; actual {tim.get('actual_minutes',float('nan')):.1f} min including GPU queue; NVIDIA L4; {stop['estimated_cu']:.3f} CU estimated from session wall time at {stop['rate_cu_per_hour']:.2f} CU/hour."
rows=['| Dataset | System | nDCG@10 % [95% CI] | Recall@100 % [95% CI] |','|---|---|---:|---:|']
for d in DS:
 for s in SYS:
  z=a['datasets'][d]['systems'][s];rows.append(f"| {d} | {s} | {fmt(z['ndcg10'])} | {fmt(z['recall100'])} |")
rows+=['','Paired differences in percentage points:','','| Dataset | Queries | Strongest single | Qwen hybrid minus strongest, nDCG@10 [95% CI] |','|---|---:|---|---:|']
for d in DS:
 z=a['datasets'][d];rows.append(f"| {d} | {z['n']} | {z['best_single']} | {fmt(z['hybrid_qwen_minus_best'],True)} |")
rows.append(f"| Equal-weight pooled | 2377 | strongest within each dataset | {fmt(a['primary'],True)} |")
tokens={s:sum(json.loads((R/d/(s+'_cost.json')).read_text())['input_tokens_truncated_nonpadding'] for d in ['scifact']+DS) for s in ['bge','qwen']}
content=f'''# Pilot-tuned hybrid search on fresh BEIR datasets

**Verdict: {a['verdict']}** under the pre-registered practical-margin rule.

**Headline number: {headline}.** Primary paired 95% CI: {fmt(a['primary'],True)} pp. PROVEN requires the CI lower endpoint >=+2 pp; DISPROVEN means the upper endpoint is below +2 pp; otherwise INCONCLUSIVE. The verdict concerns this margin on these three datasets, not the universal usefulness of fusion.

{timing}

## Results

'''+ '\n'.join(rows)+f'''

The dataset intervals use the observed strongest comparator fixed within that dataset. The primary pooled interval repeats strongest-system selection in every bootstrap draw. All intervals use 10,000 paired within-dataset query resamples, seed 12020261018. Query-weighted pooled gain, secondary: {fmt(a['query_weighted_gain'],True)} pp. The equal-weight headline averages dataset-specific gains, so it need not equal hybrid minus the largest single column average.

## Setup and tuning

SciFact (300 pilot queries) was used in #120 and excluded from confirmation. Fresh confirmation used all 648 FiQA, 323 NFCorpus and 1406 ArguAna test queries against full corpora of 57,638, 3,633 and 8,674 documents. HF data and model revisions are saved. BM25 used rank_bm25 defaults and regex word tokens; dense models were BAAI/bge-small-en-v1.5 and Qwen/Qwen3-Embedding-0.6B, exact normalized cosine search and 512-token limits. BGE used float32, Qwen float16. Query instructions and all grids were committed before scoring. ArguAna self matches were excluded identically.

Pilot choices: BGE k={sel['bge']['k']}, dense weight={sel['bge']['w']}, pilot nDCG={sel['bge']['ndcg10']*100:.2f}%; Qwen k={sel['qwen']['k']}, dense weight={sel['qwen']['w']}, pilot nDCG={sel['qwen']['ndcg10']*100:.2f}%. The full 40-row tuning grid is in results/pilot_grid.json. Fusion uses ranks through 1000 per component. BM25 ranks were precomputed on local CPU while waiting, a scheduling deviation disclosed before confirmatory metrics; no fresh metric affected selection.

## Costs, audit and limits

NVIDIA L4. Model tokenizer counts across pilot plus confirmation: BGE {tokens['bge']:,} and Qwen {tokens['qwen']:,} non-padding input tokens after truncation; generated tokens 0. Per-dataset cost JSON files split corpus/query encoding and search timing. These are batch/cold-run measurements, not production p95 request latency. BM25 wall times include local corpus indexing and query ranking; its environment is saved separately.

Query-level uncertainty conditional on three fixed datasets does not measure uncertainty across future domains. HF benchmarks may have appeared in model training. Basic unstemmed BM25 is not a tuned Lucene baseline. The 512-token cap constrains Qwen's longer-context capability and the two tokenizers do not preserve identical spans. Five ArguAna gold document IDs are absent from the HF corpus; the corresponding five queries remain included with zero scores for every system, as required by the all-query protocol. Relevance judgements are incomplete; retrieval metrics do not measure generated-answer quality. The fusion grid required at least 20% lexical weight and excluded the standalone endpoints, which were scored as baselines rather than eligible fusion settings. Forty pilot configurations can overfit 300 queries; fresh confirmation is the safeguard. Corpus and query embeddings are checkpointed on the VM during the run; small embedding audit samples, complete retrieval ranks, qrels and per-query scores are saved locally. Full embeddings are not required to recompute the reported metrics.

## Reproduce

After sourcing /workspace/tools/setup.sh, use /workspace/tools/mlenv/bin/python. scripts/prepare.py fetches public HF data using saved revisions. scripts/bm25_cpu.py computes resumable lexical ranks. scripts/provision.py checks named-session state and waits for allocation, uploads inputs and starts the detached worker. scripts/monitor.py downloads snapshots, monitors CU and stops only b9-120b on completion/failure. If a session already exists, inspect scripts/resume_probe.py output before restarting a worker. Local reproduction of statistics and charts: python scripts/check_metrics.py; python scripts/audit_results.py; python scripts/analyze.py; python scripts/report.py. No push or paid external API.

Figures: figures/scores.png, figures/gains.png, figures/recall.png, figures/table.png and the rendered figures/diagram.png. Terminal GPU identity and timing evidence: figures/colab_run.txt. Preregistration commit: af27633.

Sources: [RRF paper](https://doi.org/10.1145/1571941.1572114), [BGE model](https://huggingface.co/BAAI/bge-small-en-v1.5), [Qwen model](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [FiQA](https://huggingface.co/datasets/BeIR/fiqa), [NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus), [ArguAna](https://huggingface.co/datasets/BeIR/arguana).
'''
takeaway = (f"On three fresh search benchmarks, pilot-tuned keyword-plus-Qwen retrieval changed ranking quality by {a['primary']['mean']*100:+.2f} percentage points relative to the strongest standalone retriever. "
             + {'PROVEN': 'The result cleared our preregistered two-point improvement threshold.', 'DISPROVEN': 'The confidence interval ruled out the preregistered two-point improvement on this benchmark mix.', 'INCONCLUSIVE': 'The confidence interval did not resolve whether the improvement cleared our two-point threshold.'}[a['verdict']])
content += '\n## Two-sentence takeaway\n\n' + takeaway + '\n'
pilot=json.loads((R/'pilot_single_metrics.json').read_text())
content += f"\nPilot standalone nDCG@10: BM25 {pilot['bm25']['ndcg10']*100:.2f}%, BGE {pilot['bge']['ndcg10']*100:.2f}%, Qwen {pilot['qwen']['ndcg10']*100:.2f}%. The tuned Qwen hybrid's pilot gain over the strongest single was {(sel['qwen']['ndcg10']-max(v['ndcg10'] for v in pilot.values()))*100:.2f} pp. These are development scores, excluded from confirmation; unlike #120 this run adds model-specific query instructions.\n"
content += '\nElapsed timing includes the allocation queue and an interruption between attempts; stage accounting is in results/timing.json. The exact executed worker is retained as results/executed_worker.py; scripts/run.py additionally makes checkpoint writes atomic for subsequent reruns.\n'
Path('writeup.md').write_text(content)
p=Path('README.md');old=p.read_text().split('<!-- EXECUTION_RESULTS -->')[0]
p.write_text(old+f"\n<!-- EXECUTION_RESULTS -->\n## Measured execution\n\nVerdict: **{a['verdict']}**. Primary gain {fmt(a['primary'],True)} pp.\n\n{timing}\n\nPreregistration commit af27633 precedes all retrieval/evaluation. See writeup.md for full CIs and caveats, blog.md for the publication draft, results/analysis.json for statistics and figures/colab_run.txt for the GPU run log. GPU allocation/stop/usage records are under results/. CU is a rate-times-duration estimate because account balance also includes concurrent experiments.\n")
print(headline)
