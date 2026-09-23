# Design: FAST V1 Production Acceleration and Final Submission Pipeline

Date: 2026-09-23
Project: Chinese hard-subtitle detection on MP4
Primary detector: FAST-B736
Primary goal: maximum end-to-end speed without reducing visual box quality
Execution order: RTX 3070 server -> Git -> Quyt CPU validation

## 1. Outcome

The final submission pipeline shall:
- accept readable H.264/H.265 MP4 input at common 720p-1080p resolutions;
- scan a configurable bottom ROI covering 25%-45% of frame height (default 45%);
- detect hard-coded Chinese subtitle regions;
- produce one tight bounding box per subtitle line;
- maintain temporal stability without visible flicker;
- preserve original frame rate, dimensions, and audio;
- export a boxed MP4 for visual review;
- export timestamped per-frame bounding-box JSON;
- report measured processing time, FPS, and latency.

Text transcription is not a required output for this version.

## 2. Assignment interpretation

The mandatory deliverables are the boxed MP4 and performance report.
Coordinate JSON is the selected machine-readable output.
SRT generation is outside the final production path because transcription
is not required for the requested deliverable.
The production path is therefore detection-only:
MP4 -> ROI -> FAST -> line grouping -> temporal reconstruction
-> tight geometry refinement -> render -> MP4 + JSON.

No OCR recognizer is allowed on the critical production path.
The existing Chinese recognizer code may remain available for offline diagnosis,
but it must be disabled by default and excluded from benchmark timing.

## 3. Golden quality reference

The current FAST temporal V1 output is the regression reference.
Performance work is accepted only when final visual quality is not worse.

Quality invariants:
- no new visible subtitle misses;
- no new false subtitle boxes;
- no one- or two-frame flicker;
- no ghost boxes after a subtitle disappears;
- no merging of two subtitle lines into one block;
- no clipping of visible glyph strokes or subtitle outline;
- no increase in line overlap;
- stable box edges for static subtitle segments.

The current model/checkpoint remains FAST-B736.
Model replacement is not part of this optimization phase.

## 4. Tight-box geometry

Raw FAST boxes are not rendered directly.
Final line geometry is produced from temporal evidence.
The geometry flow is:
FAST component -> line grouping -> subtitle track -> canonical stable box
-> glyph-content refinement -> small safety margin -> final rendered box.

The refinement must:
- use evidence from multiple frames in the same subtitle segment;
- ignore short detector outliers;
- preserve separate rows in multiline subtitles;
- avoid frame-by-frame shrinking and expanding;
- retain a configurable safety margin of a few pixels;
- never use text recognition as a prerequisite.

Visual regression on the supplied test video is mandatory.

## 5. Device model

The CLI shall support:
- --device auto
- --device cuda
- --device cpu

auto chooses CUDA when a compatible NVIDIA GPU is available,
otherwise it chooses CPU.

The same detector checkpoint, thresholds, ROI, temporal logic, and final
geometry rules are used on CPU and GPU so benchmark comparisons remain valid.

## 6. GPU acceleration

RTX 3070 is the first implementation and benchmark target.

GPU optimization candidates:
- torch.inference_mode();
- optional FP16/autocast;
- cuDNN benchmark after fixed input shape is known;
- batched ROI inference;
- pinned host memory where useful;
- non-blocking host-to-device copies;
- preallocated reusable tensors/buffers where practical;
- adaptive batch-size selection;
- warm-up before timed benchmark;
- avoid unnecessary CUDA synchronization;
- overlap decode/preprocess with GPU inference where safe.

Every optimization must be benchmarked against the FP32 golden path.
FP16 is retained only if box regression remains within the accepted tolerance.

## 7. CPU acceleration

Quyt is the first CPU validation target.

CPU mode shall:
- avoid all CUDA-only calls;
- use torch/oneDNN CPU inference where supported;
- tune intra-op thread count;
- avoid oversubscription between decoder and inference workers;
- benchmark practical batch sizes;
- reuse preprocessing buffers when possible;
- parallelize decode/preprocess only when it improves end-to-end throughput.

The measured result on Quyt is reported as CPU evidence, not estimated data.

## 8. Pipeline concurrency

The preferred production data flow is:
decoder/preprocessor -> bounded ROI queue -> detector batches
-> temporal/postprocess -> renderer/encoder.

Queues must be bounded to prevent unbounded RAM use.
Concurrency must preserve input frame order exactly.
If concurrency introduces nondeterministic geometry or frame loss,
the simpler deterministic path wins even if it is slightly slower.

## 9. Output contract

Every successful run produces:
1. final boxed MP4;
2. final JSON;
3. benchmark/metrics JSON or log.

The MP4 must:
- preserve source resolution;
- preserve source FPS;
- preserve audio when present;
- support H.264 output by default and H.265 when explicitly requested;
- contain boxes on every frame where a subtitle is active;
- draw a consistent anti-aliased line box with configurable thickness and no opaque fill;
- remain playable with a standard MP4 player.

The coordinate JSON must include every processed frame, using an empty box list for frames without active subtitles, and at least:
- source video metadata;
- frame index;
- timestamp;
- track/subtitle identifier;
- line identifier;
- bbox [x1, y1, x2, y2];
- whether geometry was observed or temporally reconstructed.

No recognized text field is required.

## 10. Performance measurements

For each benchmark run record:
- device name;
- CPU model;
- GPU model when present;
- RAM and VRAM where available;
- operating system;
- frame count and video duration;
- warm-up policy;
- batch size and precision;
- detector inference seconds;
- detector FPS;
- total processing seconds;
- end-to-end FPS;
- average latency;
- P50 and P95 latency where measurable;
- peak RAM/VRAM where practical.

The benchmark must distinguish detector throughput from full pipeline throughput.

## 11. Benchmark sequence

Phase 1: RTX 3070 server
- establish deterministic FP32 reference;
- test GPU batch sizes;
- test FP16;
- optimize decode/preprocess/inference/encode overlap;
- run full supplied video;
- export final review MP4 and JSON;
- run regression tests.

Phase 2: Git handoff
- commit implementation and tests;
- push the selected branch;
- record exact commit hash used for benchmark.

Phase 3: Quyt
- clone/pull the exact commit;
- install/reuse documented environment;
- run CPU-only benchmark on the same input video;
- export a CPU review MP4 and JSON;
- compare geometry against server golden output.

RTX 3060 and T4 measurements follow only after these two phases are stable.
## 12. Test strategy

Automated tests must cover:
- explicit cpu selection;
- explicit cuda selection;
- auto fallback behavior;
- detection-only path with recognizer never initialized;
- deterministic frame count and timestamp order;
- JSON output schema;
- audio-preserving final MP4;
- one-line and two-line subtitle geometry;
- one/two-frame internal gap reconstruction;
- no ghost box at real subtitle endings;
- CPU/GPU geometry comparison within tolerance;
- FP32/FP16 geometry comparison within tolerance.

Real-video validation must include the complete supplied test video.

## 13. Acceptance criteria

Implementation is ready for submission only when:
1. FAST V1 remains the detector and quality reference.
2. Production mode performs no OCR recognition.
3. Bottom ROI is configurable from 25%-45% and defaults to 45%.
4. H.264/H.265 MP4 input and 720p/1080p handling are verified.
5. Final boxes are visibly tight around each subtitle line.
6. Static subtitle segments have no visible edge jitter.
7. One/two-frame detector gaps do not create flicker.
8. Real subtitle endings do not retain ghost boxes.
9. Final MP4 preserves source timing, dimensions, and audio.
10. H.264 output is verified; H.265 output remains available as an option.
11. JSON contains valid per-frame timestamped bbox output.
12. RTX 3070 full-video benchmark is measured and reproducible.
13. Quyt CPU full-video benchmark is measured and reproducible.
14. Optimized output is not visually worse than the golden V1 output.
15. Git contains the exact tested implementation and documented run command.
16. Final artifacts are suitable for the requested performance report.

## 14. Non-goals

- subtitle transcription;
- per-frame OCR recognition;
- SRT generation for the final submission;
- training a new detector;
- switching the primary detector to PP-OCRv6;
- changing the problem into real-time streaming;
- sacrificing box quality solely to increase FPS.
