# 08 (120b): Pilot already failed the margin in-sample

- **Experiment:** 120b (`120b-hybrid-search-fresh/`)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `120b-hybrid-search-fresh.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**Pilot already failed the margin in-sample:** the tuned Qwen hybrid beat the best single (BGE 71.27) by only **1.47 pp on SciFact itself**, below the 2 pp target even with tuning on the same queries. The hypothesis was very likely to fail before confirmation. The blog admits this ("The pilot gave reason for caution").

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
