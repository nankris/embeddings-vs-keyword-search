# 02 (120): Mid-run protocol changes

- **Experiment:** 120 (`120-embeddings-vs-keyword-search/`)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `120-embeddings-vs-keyword-search.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**Mid-run protocol changes** (BGE added, "primary fixed to BGE", the hybrid criterion reinterpreted) after a MiniLM-only run. MiniLM alone would have *failed* the 5-point gain (-0.72). Choosing BGE as primary after the MiniLM run existed looks like a forking path, even if MiniLM scores weren't viewed. Unverifiable.

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
