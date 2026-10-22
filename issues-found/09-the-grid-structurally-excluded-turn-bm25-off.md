# 09: The grid structurally excluded "turn BM25 off"

- **Study:** [hybrid search on fresh datasets](../hybrid-search-fresh/)
- **Found by:** an independent read-only review of this repo on 21 Oct 2026. Numbers and paths below were checked against the repo's results files.

## Issue and evidence
**The grid structurally excluded "turn BM25 off"** (w <= 0.8). With a much weaker BM25 on fresh sets (FiQA 21.7 vs Qwen 46.9), forced 50% BM25 weight plus a sharp k=10 is predictably harmful. The result mostly shows that **SciFact is a BM25-friendly tuning set** (BM25 65 vs dense ~71) and doesn't transfer, rather than that "hybrid search doesn't work". The blog's production advice does note this.

## Status
Open, documented. Nothing in the results was changed because of it. The README and writeup state where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data, not an edit to these numbers.
