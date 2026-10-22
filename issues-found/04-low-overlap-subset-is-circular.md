# 04: Low-overlap subset is circular

- **Study:** [original run](../original-run/)
- **Found by:** an independent read-only review of this repo on 21 Oct 2026. Numbers and paths below were checked against the repo's results files.

## Issue and evidence
**Low-overlap subset is circular:** selecting queries by low lexical overlap with the gold doc disadvantages BM25 by construction, so the +13.8 is expected and shouldn't be headlined. n=30 with a CI from +0.8 to +28.

## Status
Open, documented. Nothing in the results was changed because of it. The README and writeup state where it changes how the result should be read. Any fix needs a new pre-registered run on fresh data, not an edit to these numbers.
