# 04 (120): Low-overlap subset is circular

- **Experiment:** 120 (`120-embeddings-vs-keyword-search/`)
- **Found by:** an independent read-only review of the repo on 21 Oct 2026, 21:13-21:18 IST (fact sheet `120-embeddings-vs-keyword-search.md`). Numbers and paths below come from that review, which checked them against the repo's results files.

## Issue and evidence
**Low-overlap subset is circular:** selecting queries by low lexical overlap with the gold doc disadvantages BM25 by construction, so the +13.8 is expected and shouldn't be headlined. n=30 with a CI from +0.8 to +28.

## Status
Open, documented. Nothing in the results was changed because of it. The README/writeup or the Substack post states where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data (hypothesis rules), not an edit to these numbers.
