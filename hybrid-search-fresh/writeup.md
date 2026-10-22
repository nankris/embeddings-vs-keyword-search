# Pilot-tuned hybrid search on fresh BEIR datasets

**Verdict: DISPROVEN** under the pre-registered practical-margin rule.

**Headline number: -6.85 nDCG@10 percentage points versus the strongest single retriever (equal dataset weights).** Primary paired 95% CI: -6.85 [-7.61, -6.10] pp. PROVEN requires the CI lower endpoint >=+2 pp; DISPROVEN means the upper endpoint is below +2 pp; otherwise INCONCLUSIVE. The verdict concerns this margin on these three datasets, not the universal usefulness of fusion.

Timing: estimated 100 min; actual 229.1 min including GPU queue; NVIDIA L4; 0.773 CU estimated from session wall time at 1.54 CU/hour.

## Results

| Dataset | System | nDCG@10 % [95% CI] | Recall@100 % [95% CI] |
|---|---|---:|---:|
| fiqa | bm25 | 21.67 [19.38, 23.94] | 47.37 [44.28, 50.56] |
| fiqa | bge | 40.35 [37.48, 43.19] | 69.63 [66.76, 72.43] |
| fiqa | qwen | 46.87 [44.09, 49.65] | 79.61 [77.17, 81.99] |
| fiqa | hybrid_bge | 40.57 [37.69, 43.40] | 70.29 [67.46, 73.05] |
| fiqa | hybrid_qwen | 39.66 [37.01, 42.27] | 75.31 [72.66, 77.91] |
| nfcorpus | bm25 | 30.67 [27.38, 34.05] | 23.68 [20.81, 26.54] |
| nfcorpus | bge | 34.38 [30.85, 37.86] | 31.11 [28.13, 34.11] |
| nfcorpus | qwen | 36.14 [32.65, 39.65] | 34.03 [30.96, 37.08] |
| nfcorpus | hybrid_bge | 35.44 [31.88, 38.93] | 32.45 [29.39, 35.50] |
| nfcorpus | hybrid_qwen | 35.71 [32.35, 39.14] | 32.81 [29.74, 35.90] |
| arguana | bm25 | 38.69 [36.79, 40.64] | 84.07 [82.15, 85.92] |
| arguana | bge | 60.57 [58.73, 62.45] | 98.79 [98.22, 99.29] |
| arguana | qwen | 67.74 [66.03, 69.41] | 99.50 [99.08, 99.86] |
| arguana | hybrid_bge | 59.10 [57.30, 60.92] | 98.86 [98.29, 99.36] |
| arguana | hybrid_qwen | 54.83 [53.03, 56.67] | 99.43 [99.00, 99.79] |

Paired differences in percentage points:

| Dataset | Queries | Strongest single | Qwen hybrid minus strongest, nDCG@10 [95% CI] |
|---|---:|---|---:|
| fiqa | 648 | qwen | -7.21 [-8.69, -5.74] |
| nfcorpus | 323 | qwen | -0.43 [-1.64, +0.76] |
| arguana | 1406 | qwen | -12.90 [-14.20, -11.58] |
| Equal-weight pooled | 2377 | strongest within each dataset | -6.85 [-7.61, -6.10] |

The dataset intervals use the observed strongest comparator fixed within that dataset. The primary pooled interval repeats strongest-system selection in every bootstrap draw. All intervals use 10,000 paired within-dataset query resamples, seed 12020261018. Query-weighted pooled gain, secondary: -9.66 [-10.54, -8.75] pp. The equal-weight headline averages dataset-specific gains, so it need not equal hybrid minus the largest single column average.

## Setup and tuning

SciFact (300 pilot queries) was used in [the original run](../original-run/) and excluded from confirmation. Fresh confirmation used all 648 FiQA, 323 NFCorpus and 1406 ArguAna test queries against full corpora of 57,638, 3,633 and 8,674 documents. HF data and model revisions are saved. BM25 used rank_bm25 defaults and regex word tokens; dense models were BAAI/bge-small-en-v1.5 and Qwen/Qwen3-Embedding-0.6B, exact normalized cosine search and 512-token limits. BGE used float32, Qwen float16. Query instructions and all grids were committed before scoring. ArguAna self matches were excluded identically.

Pilot choices: BGE k=10, dense weight=0.8, pilot nDCG=72.42%; Qwen k=10, dense weight=0.5, pilot nDCG=72.74%. The full 40-row tuning grid is in results/pilot_grid.json. Fusion uses ranks through 1000 per component. BM25 ranks were precomputed on local CPU while waiting, a scheduling deviation disclosed before confirmatory metrics; no fresh metric affected selection.

## Costs, audit and limits

NVIDIA L4. Model tokenizer counts across pilot plus confirmation: BGE 14,733,875 and Qwen 14,538,930 non-padding input tokens after truncation; generated tokens 0. Per-dataset cost JSON files split corpus/query encoding and search timing. These are batch/cold-run measurements, not production p95 request latency. BM25 wall times include local corpus indexing and query ranking; its environment is saved separately.

Query-level uncertainty conditional on three fixed datasets does not measure uncertainty across future domains. HF benchmarks may have appeared in model training. Basic unstemmed BM25 is not a tuned Lucene baseline. The 512-token cap constrains Qwen's longer-context capability and the two tokenizers do not preserve identical spans. Five ArguAna gold document IDs are absent from the HF corpus; the corresponding five queries remain included with zero scores for every system, as required by the all-query protocol. Relevance judgements are incomplete; retrieval metrics do not measure generated-answer quality. The fusion grid required at least 20% lexical weight and excluded the standalone endpoints, which were scored as baselines rather than eligible fusion settings. Forty pilot configurations can overfit 300 queries; fresh confirmation is the safeguard. Corpus and query embeddings are checkpointed on the VM during the run; small embedding audit samples, complete retrieval ranks, qrels and per-query scores are saved locally. Full embeddings are not required to recompute the reported metrics.

## Reproduce

Install the packages listed in the imports of the scripts (numpy, sentence-transformers, rank_bm25, pyarrow, pandas). scripts/prepare.py fetches public HF data using saved revisions. scripts/bm25_cpu.py computes resumable lexical ranks. scripts/provision.py checks named-session state and waits for allocation, uploads inputs and starts the detached worker. scripts/monitor.py downloads snapshots, monitors CU and stops only hybrid-search on completion/failure. If a session already exists, inspect scripts/resume_probe.py output before restarting a worker. Local reproduction of statistics and charts: python scripts/check_metrics.py; python scripts/audit_results.py; python scripts/analyze.py; python scripts/report.py. No push or paid external API.

Figures: figures/scores.png, figures/gains.png, figures/recall.png, figures/table.png and the rendered figures/diagram.png. Terminal GPU identity and timing evidence: figures/colab_run.txt. Preregistration commit: af27633.

Sources: [RRF paper](https://doi.org/10.1145/1571941.1572114), [BGE model](https://huggingface.co/BAAI/bge-small-en-v1.5), [Qwen model](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [FiQA](https://huggingface.co/datasets/BeIR/fiqa), [NFCorpus](https://huggingface.co/datasets/BeIR/nfcorpus), [ArguAna](https://huggingface.co/datasets/BeIR/arguana).

## Two-sentence takeaway

On three fresh search benchmarks, pilot-tuned keyword-plus-Qwen retrieval changed ranking quality by -6.85 percentage points relative to the strongest standalone retriever. The confidence interval ruled out the preregistered two-point improvement on this benchmark mix.

Pilot standalone nDCG@10: BM25 65.19%, BGE 71.27%, Qwen 70.09%. The tuned Qwen hybrid's pilot gain over the strongest single was 1.47 pp. These are development scores, excluded from confirmation; unlike the original run, this one adds model-specific query instructions.

Elapsed timing includes the allocation queue and an interruption between attempts; stage accounting is in results/timing.json. The exact executed worker is retained as results/executed_worker.py; scripts/run.py additionally makes checkpoint writes atomic for subsequent reruns.
