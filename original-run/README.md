# Embeddings / semantic search vs BM25 keyword search

## Concept in one sentence
An **embedding** turns text into a vector of numbers so that "search" can match on *meaning*
(vector closeness) instead of on shared *words* (keyword/BM25 overlap).

## Hypothesis
On a small public retrieval benchmark, a small sentence-embedding model beats BM25 keyword
search by **>= 5 nDCG@10 points**, and a **hybrid** (reciprocal rank fusion of BM25 + dense)
beats both individually.

## Setup
- Data: public **BEIR SciFact** (HF `BeIR/scifact`, `BeIR/scifact-qrels`). Corpus 5,183 abstracts;
  test queries 300 (1,109 public source queries encoded, only test queries scored; paired qrels, avg ~1.13 relevant docs/query, score>0 = relevant).
- Systems
  - `bm25` — `rank_bm25.BM25Okapi` over lowercased alphanumeric regex tokens (`[a-z0-9]+`) of `title + text`.
  - `dense_minilm` — `sentence-transformers/all-MiniLM-L6-v2` (384-d), cosine, top-10.
  - `dense_bge`  — `BAAI/bge-small-en-v1.5` (384-d), cosine, top-10.
  - Dense defaults: no manually added query prefix; MiniLM max input 256 tokens, BGE 512 tokens. Corpus text is title + abstract.
  - `hybrid` — reciprocal rank fusion (RRF, k=60) of `bm25` + `dense_minilm`.
- Metrics: **nDCG@10** and **Recall@10** (binary relevance from qrels), per query.
- Uncertainty: **paired bootstrap** over queries (10,000 resamples) → 95% percentile CI on the
  mean paired difference. Paired = same resampled query set for both systems.
- Low-overlap diagnostic subset (original queries, not authored paraphrases): 30 SciFact test queries with the **lowest Jaccard lexical overlap** between
  the query and its relevant abstract (i.e. query shares few words with the answer). Built cheaply
  from data we already have — no LLM needed.
- Hardware: **CPU only** (aarch64, 2 cores, ~7 GB RAM). No GPU, no Colab → **0 CU used**.

## Fixed decision rule (set BEFORE running)
Using nDCG@10, paired bootstrap over queries, 10,000 resamples, 95% CI:

1. **PROVEN** — (`dense_bge` − `bm25`) 95% CI lower bound **> 0** AND point estimate **>= +0.05**;
   and (`hybrid` − `max(bm25, dense)`) 95% CI lower bound **> 0**.
2. **DISPROVEN** — (`dense_bge` − `bm25`) 95% CI **contains 0**, i.e. no reliable embedding gain.
3. **INCONCLUSIVE** — otherwise (CI excludes 0 but point gain < 5 points, or the two parts disagree).

## Headline number
"Embeddings found **X% more right answers** than keyword search." — X from Recall@10
(relative increase of dense over BM25), with the paraphrase subset as the secondary angle.

## Notes recorded on 2026-10-16, before scoring
- Reused the existing data and MiniLM embedding caches; preserved the original decision rule and BM25+MiniLM hybrid.
- Interpret hybrid superiority as positive lower bounds against each of BM25, MiniLM and BGE (no per-query oracle).
- Primary embedding and headline are fixed to BGE; do not pick a model after seeing scores.
- The 30-query subset is an exploratory **low lexical overlap** diagnostic, not independently authored paraphrases. Jaccard can be small simply because abstracts are long.
- The decision rule is retained verbatim: its DISPROVEN category means failure of its significance test, not proof of no effect.

## Status
- Status: complete — see writeup.md for results and chart.png for the comparison.

## Running and artifacts
Use a Python environment with `scripts/requirements.txt` installed for `scripts/01_prepare_data.py`, `scripts/02_run_retrieval.py`, `scripts/04_audit_cache.py`, then `scripts/03_report.py`. The retrieval script checkpoints each model’s embeddings and rankings; the report can be regenerated from ranks without inference. Inputs, cached vectors, top-100 ranks, per-query CSV, bootstrap JSON and a hash/software manifest are kept in `results/`. Model loading uses public Hugging Face downloads, with no paid API.

## Measured outcome
- Overall verdict: **INCONCLUSIVE**: BGE passes the embedding-gain rule, but the fixed BM25+MiniLM hybrid does not beat BGE.
- BGE versus BM25: nDCG@10 **72.00 vs 65.23**, difference **+6.77 points** (95% paired CI **+3.04 to +10.55**).
- Recall@10: **84.52% vs 77.57%**, or **9.0% more right answers** on the specified macro-recall definition. The 95% paired-bootstrap relative-gain CI is **3.2% to 15.4%**.
- BGE corpus encoding/model import took 1,241.4 seconds and query encoding/model reload 26.7 seconds in the resumed run (saved MiniLM work excluded); see results/resume_run.log. GPU: none; Colab: unused; **CU: 0**.
- Model revisions are pinned in the scripts and recorded in results/model_sources.json.
