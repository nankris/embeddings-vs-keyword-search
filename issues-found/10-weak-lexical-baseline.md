# 10: Weak lexical baseline

- **Study:** [hybrid search on fresh datasets](../hybrid-search-fresh/)
- **Found by:** an independent read-only review of this repo on 21 Oct 2026. Numbers and paths below were checked against the repo's results files.

## Issue and evidence
Weak lexical baseline (no stemming, rank_bm25 defaults) handicaps the hybrid specifically. A Lucene/BM25+ baseline could change the magnitude, though probably not the sign on ArguAna/FiQA.

## Status
Open, documented. Nothing in the results was changed because of it. The README and writeup state where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data, not an edit to these numbers.
