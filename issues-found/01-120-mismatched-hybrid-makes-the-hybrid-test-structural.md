# 01 (120): Mismatched hybrid makes the hybrid test structurally weak

- **Experiment:** 120 (`120-embeddings-vs-keyword-search/`)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `120-embeddings-vs-keyword-search.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**Mismatched hybrid makes the hybrid test structurally weak:** the hybrid fuses BM25 with **MiniLM** (the weaker dense model) but must beat **BGE**. A BM25+BGE hybrid was never run. The INCONCLUSIVE verdict says little about hybrids; it's an artefact of component choice, which happened partly because BGE was added mid-run.

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
