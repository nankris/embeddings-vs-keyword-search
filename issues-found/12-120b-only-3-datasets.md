# 12 (120b): Only 3 datasets

- **Experiment:** 120b (`120b-hybrid-search-fresh/`)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `120b-hybrid-search-fresh.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
Only 3 datasets; the CIs cover query sampling, not domain sampling (disclosed). Contamination: BEIR/MTEB sets are standard eval sets and very likely seen in embedding-model training or selection (disclosed as "may have appeared"). 5 ArguAna queries have gold docs missing from the HF corpus and score 0 for all systems (disclosed, negligible).

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
