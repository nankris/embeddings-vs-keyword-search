# Does pilot-tuned hybrid search transfer to fresh datasets?

## Question and evidence
Does weighted Reciprocal Rank Fusion tuned on SciFact beat the better single retriever by at least 2 percentage points of nDCG@10 on fresh BEIR datasets? [The original run](../original-run/) used SciFact and found BGE-small 72.00 versus BM25 65.23 nDCG points; its fixed BM25+MiniLM hybrid scored 68.40. This motivates better components and pilot tuning, without claiming an established 2-point effect.

Source: Cormack, Clarke and Buettcher, [Reciprocal rank fusion outperforms Condorcet and individual rank learning methods](https://doi.org/10.1145/1571941.1572114), SIGIR 2009. The paper supports the direction of the prediction, not its exact magnitude or transfer to modern embedding models. [Qwen model card](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B).

## Preregistered hypothesis and fixed decision rule
Primary hypothesis: SciFact-tuned BM25+Qwen RRF improves equal-dataset-weighted mean nDCG@10 by >=0.02 over the strongest single system per fresh dataset, selected from BM25, BGE-small and Qwen. The 2-point margin is a practical threshold I set for this experiment: smaller gains rarely survive a dataset change, and running two retrievers costs more.

PROVEN if the lower endpoint of the primary paired 95% bootstrap CI is >=0.02. DISPROVEN if its upper endpoint is <0.02. Otherwise INCONCLUSIVE. This tests a practical margin, not merely superiority to zero. Report all dataset results even if pooled result fails; no requirement that every dataset pass. Dataset-level claims are descriptive, with no multiple-testing-adjusted significance claims. Secondary: tuned BM25+BGE hybrid, Recall@100, query-weighted pooled result, latency and token counts. No reranker.

## Frozen setup before confirmatory scoring
Pilot: all 300 SciFact test queries, reused only for tuning, never included in confirmatory scores. Tune k in {10,30,60,100} and dense weight w in {0.20,0.35,0.50,0.65,0.80}, separately for BGE and Qwen, maximizing pilot mean nDCG@10. Ties follow grid order. Score = w/(k+dense_rank)+(1-w)/(k+bm25_rank); absent ranks contribute zero. Fuse top 1000 per component (or full corpus if smaller). Save tuning scores and selected settings before confirmatory retrieval starts. No tuning on fresh data.

Confirmation: all positive-qrel test queries in public HF BeIR/fiqa, BeIR/nfcorpus, BeIR/arguana, full corpora. Expected n=648,323,1406, respectively (2377 total); use actual HF test count and disclose differences. These datasets were not used in the original run. Models: BAAI/bge-small-en-v1.5 and Qwen/Qwen3-Embedding-0.6B, normalized cosine similarity, float32 BGE / float16 Qwen, max 512 tokens each. BGE query prefix 'Represent this sentence for searching relevant passages: '; Qwen query prefix 'Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: '. Documents title+text, no prompt. BM25Okapi rank_bm25 defaults, lowercased regex word tokens, no stemming/stopword removal. This is a reproducible basic lexical baseline, not a tuned Lucene baseline.

nDCG@10 uses linear qrel gains and log2 discounts; Recall@100 counts positive judgements. Self matches excluded for ArguAna (query ID equals document ID); all methods use the same exclusion. Ties break by document index. 10,000 bootstrap draws, seed 12020261018, paired within each dataset. Primary equal-weight mean of three dataset gains. In each draw recompute the best single system's mean within each dataset, avoiding treating an outcome-selected comparator as fixed. Also report fixed observed-comparator differences by dataset. Query-weighted pooling is secondary, as ArguAna otherwise dominates.

Sample size is all available queries, not a post-hoc stopping rule. If paired SD is 0.20, equal-dataset SE is approximately 0.005 under independent queries, so a 95% half-width is about 1 point. To establish a >=2-point margin needs an observed gain comfortably above 2 points; dataset estimates may remain imprecise. This is a planning sensitivity calculation, not an evidence-based effect forecast. No extra seeds: retrieval is deterministic; uncertainty comes from queries. Cross-dataset generalization uncertainty is not captured by three fixed datasets.

## Execution and audit
Use a named Colab session (hybrid-search), L4, hard budget 6 CU. Run detached with nohup, checkpoint embeddings by batch and rankings by method/dataset, download results frequently. Public HF datasets only, no paid API. Scripts must inspect existing results. Model tokenizer counts of actual non-padding, truncated inputs are saved; no generated tokens (embedding workload).

Timing: estimated 100 min (15 setup, 65 retrieval/evaluation, 20 writeup), planned L4; actual and CU pending. Waiting for a free GPU slot is included in wall time and separately disclosed. Start timestamp is in results/timing.json.

## Execution scheduling note (before confirmatory metrics)
While waiting for a GPU slot, deterministic BM25 ranks were precomputed on the local aarch64 CPU. This advances lexical retrieval relative to the planned pilot-first retrieval order, but no fresh-dataset metric is scored or used for selection. Pilot tuning remains frozen before confirmatory dense retrieval and all confirmatory scoring. This changes scheduling only; hypotheses, grids, datasets, metrics and decision rule remain as committed in af27633.




<!-- EXECUTION_RESULTS -->
## Measured execution

Verdict: **DISPROVEN**. Primary gain -6.85 [-7.61, -6.10] pp.

Timing: estimated 100 min; actual 229.1 min including GPU queue; NVIDIA L4; 0.773 CU estimated from session wall time at 1.54 CU/hour.

Preregistration commit af27633 precedes all retrieval/evaluation. See writeup.md for full CIs and caveats, blog.md for the plain-language write-up, results/analysis.json for statistics and figures/colab_run.txt for the GPU run log. GPU allocation/stop/usage records are under results/. CU is a rate-times-duration estimate because account balance also includes concurrent experiments.
