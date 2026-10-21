# Embeddings versus keyword search

**Verdict: INCONCLUSIVE** under the decision rule written before scoring. BGE passes the embedding-gain criterion, but the fixed hybrid does not beat BGE.

**Headline: “Embeddings found 9.0% more right answers than keyword search.”**

This means BGE’s relative increase in mean Recall@10 versus BM25 on 300 SciFact test queries: 95% paired-bootstrap CI **3.2% to 15.4%**. A right answer here means a gold-labelled relevant abstract retrieved in the first ten results, not a generated factual answer.

| System | nDCG@10, % (95% CI) | Recall@10, % (95% CI) |
|---|---:|---:|
| BM25 | 65.23 [60.74, 69.60] | 77.57 [72.89, 82.07] |
| MiniLM | 64.51 [60.03, 69.00] | 78.33 [73.57, 82.83] |
| BGE-small | 72.00 [67.73, 76.16] | 84.52 [80.49, 88.42] |
| Hybrid (BM25 + MiniLM) | 68.40 [63.96, 72.66] | 80.59 [76.09, 84.85] |

Paired differences below are percentage points; intervals resample the same queries for each system (10,000 draws, seed 20261016).

| Comparison | Δ nDCG@10 (95% CI) | Δ Recall@10 (95% CI) |
|---|---:|---:|
| dense_bge-bm25 | +6.77 [+3.04, +10.55] | +6.96 [+2.56, +11.48] |
| dense_minilm-bm25 | -0.72 [-4.51, +3.12] | +0.77 [-3.73, +5.23] |
| hybrid-bm25 | +3.17 [+1.03, +5.39] | +3.02 [+0.34, +5.83] |
| hybrid-dense_minilm | +3.89 [+1.02, +6.73] | +2.26 [-1.77, +6.37] |
| hybrid-dense_bge | -3.60 [-6.74, -0.47] | -3.93 [-7.98, +0.08] |

## Low-overlap diagnostic (30 queries)

The original 30 test queries with the lowest maximum query/gold-abstract token Jaccard overlap were selected without using retrieval scores (maximum overlap 0.0246). These are not newly authored paraphrases; abstract length affects Jaccard, so this is only an exploratory diagnostic.

| System | nDCG@10, % (95% CI) | Recall@10, % (95% CI) |
|---|---:|---:|
| BM25 | 34.39 [18.33, 50.37] | 40.00 [23.33, 56.67] |
| MiniLM | 41.52 [26.47, 56.93] | 56.67 [40.00, 73.33] |
| BGE-small | 48.23 [32.95, 63.93] | 63.33 [46.67, 80.00] |
| Hybrid (BM25 + MiniLM) | 37.85 [21.67, 54.52] | 43.33 [26.67, 60.00] |

BGE minus BM25 on this subset: +13.84 nDCG points (95% CI +0.82 to +28.37). All subset metric intervals and paired comparisons are in `results/analysis.json`.

## Interpretation and limits

BGE’s nDCG gain is 6.77 points; the target is at least 5 points plus a positive CI lower bound. Hybrid must also have a positive nDCG CI lower bound against all three individual rankers to satisfy the retained rule. The original rule calls a CI containing zero DISPROVEN; that label means the experiment did not establish the hypothesis, not that it established equivalence.

This is one small scientific-claims benchmark, with incomplete relevance judgements and query-level uncertainty only. It does not establish performance on other domains. BM25 uses untuned rank_bm25 defaults and regex tokens, without stemming or stopword removal. Dense models use their SentenceTransformer defaults, normalized vectors, title plus abstract and no added query instruction; MiniLM truncates at its default 256 tokens and BGE at 512, so this compares practical default configurations rather than equal token budgets. Hybrid fuses only the top 100 ranks per component with k=60; no tuning or outcome-driven model selection was done. Both whole-corpus embeddings and the existing MiniLM caches are retained for audit.

## Reproduce

Run with `/workspace/tools/mlenv/bin/python`, after `source /workspace/tools/setup.sh`:

```bash
python scripts/01_prepare_data.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false python scripts/02_run_retrieval.py
python scripts/04_audit_cache.py
python scripts/03_report.py
```

Raw corpus, queries, qrels, top-100 rankings, per-query scores, caches and software/hash manifest are under `results/`. Existing embeddings and rankings are reused; remove affected cache files to recompute after changing inputs or settings. The report validates document IDs, rank uniqueness and hand-calculated metric examples. Six fixed MiniLM rows were freshly encoded and matched the resumed cache with cosine agreement above 0.9999 (`results/cache_audit.json`). CPU only; GPU: none; Colab CU: 0.

## Two-sentence takeaway

On these scientific claims, a small embedding model retrieved 9.0% more relevant papers in its first ten results than keyword search. The fixed keyword-plus-MiniLM hybrid improved on keyword search, but did not beat BGE on ranking quality.

Decision checks: BGE gain >=5 nDCG points: True; BGE gain CI excludes zero positively: True; Hybrid reliably beats bm25: True; Hybrid reliably beats dense_minilm: True; Hybrid reliably beats dense_bge: False.

Source links: [SciFact corpus and queries](https://huggingface.co/datasets/BeIR/scifact), [test relevance labels](https://huggingface.co/datasets/BeIR/scifact-qrels), [MiniLM model](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [BGE model](https://huggingface.co/BAAI/bge-small-en-v1.5).
