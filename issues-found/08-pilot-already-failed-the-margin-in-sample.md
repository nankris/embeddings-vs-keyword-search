# 08: Pilot already failed the margin in-sample

- **Study:** [hybrid search on fresh datasets](../hybrid-search-fresh/)
- **Found by:** an independent read-only review of this repo on 21 Oct 2026. Numbers and paths below were checked against the repo's results files.

## Issue and evidence
**Pilot already failed the margin in-sample:** the tuned Qwen hybrid beat the best single (BGE 71.27) by only **1.47 pp on SciFact itself**, below the 2 pp target even with tuning on the same queries. The hypothesis was very likely to fail before confirmation. The blog admits this ("The pilot gave reason for caution").

## Status
Open, documented. Nothing in the results was changed because of it. The README and writeup state where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data, not an edit to these numbers.
