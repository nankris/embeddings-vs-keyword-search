# 06: Contamination / training overlap

- **Study:** [original run](../original-run/)
- **Found by:** an independent read-only review of this repo on 21 Oct 2026. Numbers and paths below were checked against the repo's results files.

## Issue and evidence
**Contamination / training overlap:** bge-small and all-MiniLM were trained on large mixed corpora (MiniLM's 1B pairs include S2ORC/SPECTER scientific pairs). SciFact is a widely used BEIR/MTEB eval set. Absolute scores are possibly flattered; the relative comparison is less affected.

## Status
Open, documented. Nothing in the results was changed because of it. The README and writeup state where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data, not an edit to these numbers.
