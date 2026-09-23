# Reproduction validation

A clean repository copy was executed against the original 3733-frame
evaluation video using the same detector cache and V1 production command.

Validation result:

- reference coordinate records: 2581
- reproduced coordinate records: 2581
- record-by-record equality: TRUE
- frames: 3733 == 3733
- boxed frames: 2216 == 2216
- logical tracks: 52 == 52
- merged groups: 8 == 8
- merged fragmented tracks: 11 == 11
- transition fragments absorbed: 1 == 1
- temporal-glyph tightened tracks: 28 == 28
- height-regularized tracks: 7 == 7
- OCR-tightened tracks: 3 == 3
- post-lock edge-range P95: 0.0 px == 0.0 px
- post-lock max edge range: 0.0 px == 0.0 px
- multiline overlap frames: 0 == 0
- multiline overlap pixels: 0 == 0

This verifies that the committed source reproduces the exact frame-level
bounding-box coordinate stream of FAST_temporal_v1_consistent_final.json.
