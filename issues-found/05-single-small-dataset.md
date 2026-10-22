# 05: Single small dataset

- **Study:** [original run](../original-run/)
- **Found by:** an independent read-only review of this repo on 21 Oct 2026. Numbers and paths below were checked against the repo's results files.

## Issue and evidence
Single small dataset (300 queries, scientific claims). Untuned BM25 (no stemming). BGE without its query instruction (which likely understates BGE). Truncation 256 vs 512 tokens differs between the models.

## Status
Open, documented. Nothing in the results was changed because of it. The README and writeup state where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data, not an edit to these numbers.
