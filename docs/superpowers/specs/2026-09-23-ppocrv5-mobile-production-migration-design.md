# Design: PP-OCRv5 Mobile Production Migration

Date: 2026-09-23
Project: Chinese hard-subtitle detection on MP4
Supersedes detector/model-selection portions of: `2026-09-23-fast-v1-production-acceleration-design.md`
Selected production detector: `PP-OCRv5_mobile_det`

## 1. Outcome

Migrate the production detector from FAST-B736 to PP-OCRv5 Mobile Det while preserving the existing submission contract:
- input H.264/H.265 MP4, common 720p-1080p;
- configurable bottom subtitle ROI 25%-45%, default 45%;
- one tight bbox per subtitle line;
- stable boxes without visible flicker or 1-2 frame blink;
- no ghost boxes after subtitle disappearance;
- no multiline overlap;
- preserve source resolution, FPS, and audio;
- output H.264 MP4 plus per-frame timestamped bbox JSON and benchmark metrics.

Text transcription remains out of scope. The production path is detection-only.

## 2. Architecture

Production flow:

MP4 decode
-> bottom ROI crop
-> PP-OCRv5 Mobile text detection
-> score hysteresis
-> lightweight temporal track association
-> short-line recovery
-> canonical per-track bbox stabilization
-> adaptive glyph-safe padding
-> multiline seam clamp
-> render/encode
-> MP4 + JSON + metrics

No OCR recognition is required anywhere on the critical path.

## 3. Detector configuration

Default detector:
- model: `PP-OCRv5_mobile_det`;
- device: `auto|cuda|cpu`;
- ROI: bottom 45% by default;
- detector threshold: `thresh=0.30`;
- detector box threshold: `box_thresh=0.50`;
- runtime batch size is benchmark-selected per device.

The detector must only process the subtitle ROI, never the full frame unless explicitly configured for diagnosis.

## 4. Temporal hysteresis

Use two score bands:
- HIGH: score >= 0.84 starts/confirms a subtitle track;
- LOW: 0.50 <= score < 0.84 may extend/backfill a confirmed track.

LOW detections must never create arbitrary persistent boxes by themselves unless they satisfy the short-line recovery rule.

Track association uses spatial center, width/height ratio, IoU, and frame continuity. A 1-2 frame internal gap may be interpolated only when geometry before/after is consistent.

Do not fill a gap across a subtitle transition when width, height, center, or IoU indicates different content.

## 5. Short-line recovery

Single-character or very short second lines are a known failure mode of the mobile detector at default confidence.

A short-line candidate may be retained when:
- it is horizontally near the subtitle center;
- it persists across multiple frames;
- it lies below an already confirmed strong subtitle line;
- its time span substantially overlaps that strong line;
- its geometry is temporally consistent.

The retained short line gets one canonical stable bbox for the track.

## 6. Glyph-safe bbox geometry

The visual review at approximately 57 seconds exposed a clipping bug in the first PP-OCR candidate.

Observed evidence:
- previous final line-2 bbox: approximately `[346, 887, 370, 922]`;
- actual visible glyph/outline support: approximately `[340, 882, 379, 924]`.

Therefore weak/short-line canonical geometry must not use a narrow percentile box with fixed 2 px padding.

For weak/short lines:
- derive a robust union from representative high-quality frames in the same track;
- include visible white glyph cores and dark outline support;
- add adaptive safety padding proportional to glyph height, with a practical minimum padding;
- clamp the upper edge against the line-1 seam so the two line boxes never overlap;
- keep the resulting canonical box constant for the whole stable track.

Acceptance for the reviewed 57-second segment:
- no visible glyph stroke or outline is clipped;
- line 1 and line 2 remain disjoint;
- line-2 bbox does not visibly breathe frame-to-frame.

## 7. Multiline seam rule

For two simultaneous lines, compute a stable seam from canonical geometries.

The top-line bottom edge and bottom-line top edge must be clamped so that:
- there is no positive-area overlap;
- at least a 1 px separation is preserved when geometry permits;
- neither clamp cuts into visible glyph support.

The seam is track-level, not recomputed independently every frame.

## 8. Quality reference and metrics

The existing FAST temporal V1 JSON remains a regression reference for timing, track presence, and line count, but it is not absolute ground truth for tight-box area.

Cross-detector evaluation must use both:
- spatial IoU for comparable long-line boxes;
- center/baseline matching for short glyph boxes that PP-OCR may fit more tightly than FAST.

Required full-video quality checks:
- output frame count equals source;
- 0 dropped frames;
- 0 ghost subtitle frames;
- 0 multiline overlap frames;
- stabilized edge-motion median/P95 = 0 px for canonical static tracks;
- no visible clipping in representative review frames;
- short second-line around 57s remains fully enclosed.

## 9. Performance evidence

Measured RTX 3070 evidence from the supplied 3733-frame video:
- PP-OCRv5 Mobile ROI detector: approximately 48.47 detector FPS;
- decode + detector: approximately 45.73 FPS;
- lightweight temporal postprocess: sub-second for the full video in the benchmark prototype;
- H.264 yuv420p render/encode: above real-time;
- stage-sum end-to-end candidate: approximately real-time or better.

These are benchmark evidence, not hard-coded acceptance thresholds. Final report values must come from the exact committed production implementation.

## 10. Output format

Final MP4:
- H.264 High profile;
- yuv420p;
- same resolution and FPS as source;
- preserve source audio when present;
- anti-aliased bbox outline, no opaque fill.

Final JSON:
- one entry for every processed frame, including empty-box frames;
- frame index and timestamp;
- track ID and line ID;
- bbox `[x1,y1,x2,y2]`;
- source state such as HIGH, LOW_BACKFILL, GAP, or WEAK_HOLD;
- detector/runtime metadata and thresholds.

## 11. CPU/GPU behavior

GPU:
- use PaddlePaddle GPU runtime when available;
- batch ROI inference;
- warm up before timed benchmark.

CPU:
- same model, ROI, thresholds, temporal logic, and geometry rules;
- no CUDA-only calls;
- benchmark actual throughput on Quyt.

The code must expose the same CLI-level detector choice across devices.

## 12. Compatibility and dependency isolation

Production dependencies must explicitly pin a compatible PaddlePaddle/PaddleOCR environment.

FAST code may remain in the repository as a reference/fallback backend, but PP-OCRv5 Mobile becomes the default production detector after this migration.

The final benchmark report must record:
- PaddlePaddle version;
- PaddleOCR version;
- CUDA runtime when applicable;
- model name/version;
- detector thresholds;
- batch size;
- ROI;
- Git commit.

## 13. Tests

Add automated tests for:
- HIGH/LOW hysteresis;
- LOW-before-HIGH backfill;
- no gap fill across geometry-changing subtitle transitions;
- short-line persistence;
- adaptive glyph-safe padding;
- multiline seam non-overlap;
- stable canonical bbox over a track;
- CPU device fallback;
- JSON completeness;
- output frame/FPS/audio preservation where supported by the test fixture.

Add a regression fixture/test for the 57-second line-2 clipping case using representative geometry so the final box fully encloses glyph support and remains disjoint from line 1.

## 14. Rollout

1. Integrate PP-OCRv5 Mobile as a production backend.
2. Make it the default detector without deleting FAST fallback.
3. Port the benchmark prototype's light temporal logic into tested source modules.
4. Apply the glyph-safe short-line correction validated in the review clip.
5. Run targeted tests, then the full suite.
6. Render the full 3733-frame video.
7. Visually re-check representative frames, especially ~57s and subtitle transitions.
8. Record exact RTX 3070 metrics from the committed implementation.
9. Commit and push the feature branch.
10. Clone/pull the exact commit on Quyt and run CPU benchmark.

## 15. Git and submission policy

The pushed Git branch and exact commit hash are the source-code handoff.

Do not treat local benchmark venvs, downloaded model caches, or generated review videos as source files.

The hiring submission package still uses the final boxed MP4 and performance report; Git identifies the implementation that generated them.
