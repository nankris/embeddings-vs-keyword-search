# 12: Only 3 datasets

- **Study:** [hybrid search on fresh datasets](../hybrid-search-fresh/)
- **Found by:** an independent read-only review of this repo on 21 Oct 2026. Numbers and paths below were checked against the repo's results files.

## Issue and evidence
Only 3 datasets; the CIs cover query sampling, not domain sampling (disclosed). Contamination: BEIR/MTEB sets are standard eval sets and very likely seen in embedding-model training or selection (disclosed as "may have appeared"). 5 ArguAna queries have gold docs missing from the HF corpus and score 0 for all systems (disclosed, negligible).

## Status
Open, documented. Nothing in the results was changed because of it. The README and writeup state where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data, not an edit to these numbers.
