# 05 (120): Single small dataset

- **Experiment:** 120 (`120-embeddings-vs-keyword-search/`)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `120-embeddings-vs-keyword-search.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
Single small dataset (300 queries, scientific claims). Untuned BM25 (no stemming). BGE without its query instruction (which likely understates BGE). Truncation 256 vs 512 tokens differs between the models.

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
