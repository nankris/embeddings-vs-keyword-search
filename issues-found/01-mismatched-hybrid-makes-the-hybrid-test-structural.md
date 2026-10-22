# 01: Mismatched hybrid makes the hybrid test structurally weak

- **Study:** [original run](../original-run/)
- **Found by:** an independent read-only review of this repo on 21 Oct 2026. Numbers and paths below were checked against the repo's results files.

## Issue and evidence
**Mismatched hybrid makes the hybrid test structurally weak:** the hybrid fuses BM25 with **MiniLM** (the weaker dense model) but must beat **BGE**. A BM25+BGE hybrid was never run. The INCONCLUSIVE verdict says little about hybrids; it's an artefact of component choice, which happened partly because BGE was added mid-run.

## Status
Open, documented. Nothing in the results was changed because of it. The README and writeup state where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data, not an edit to these numbers.
