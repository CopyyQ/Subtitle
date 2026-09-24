# PP-OCRv5 Mobile Production Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the default FAST-B production detector with PP-OCRv5 Mobile Det plus lightweight temporal hysteresis, preserving the existing MP4/JSON contract while fixing the short second-line clipping observed around 57 seconds.

**Architecture:** Add a PP-OCR detector backend behind the existing `DetectorBackend` interface, add a dedicated lightweight PP-OCR temporal module, and select that path through `PipelineConfig.detector`. FAST remains available as an explicit fallback. Short weak lines are stabilized from multi-frame evidence, expanded to glyph support with adaptive padding, and clamped to a stable multiline seam.

**Tech Stack:** Python 3.10, OpenCV, NumPy, SciPy, PaddlePaddle 3.3.1, PaddleOCR 3.7.0, existing PyTorch/FAST fallback, pytest, ffmpeg.

**Spec:** `docs/superpowers/specs/2026-09-23-ppocrv5-mobile-production-migration-design.md`

## Global Constraints

- Default detector is `PP-OCRv5_mobile_det`.
- Bottom ROI remains configurable within 25%-45%; default is 45%.
- Detector defaults are `thresh=0.30`, `box_thresh=0.50`, HIGH score `>=0.84`, LOW score `0.50..0.84`.
- Production path is detection-only; OCR recognition must not run.
- FAST remains an explicit fallback backend.
- CPU and GPU use the same detector model, thresholds, ROI, temporal logic, and geometry rules.
- Final MP4 is H.264 High, yuv420p, same resolution/FPS as input, with source audio preserved.
- Final JSON contains every processed frame, including empty-box frames.
- No visible glyph clipping, no new subtitle miss, no ghost box, no multiline overlap, no visible flicker.
- Short-line geometry around 57 seconds must fully contain visible glyph/outline support and remain disjoint from line 1.
- Generated videos, model caches, local virtualenvs, and benchmark logs are not committed as source.

## Review Focus

- A LOW detection immediately before the first HIGH frame of a real subtitle must be backfilled rather than lost.
- A one-frame blank between two geometrically different subtitles must remain blank rather than be gap-filled.
- A one-character second line with detector boxes narrower than its visible outline must be expanded without overlapping line 1.
- A frame containing non-subtitle high-confidence text outside the learned subtitle band must not become a persistent subtitle track.
- CPU execution on a machine with CUDA libraries installed must never invoke CUDA runtime operations.

---

### Task 1: PP-OCRv5 Mobile detector backend and runtime selection

**Files:**
- Create: `src/video_text/ppocrv5_backend.py`
- Modify: `src/video_text/pipeline.py`
- Modify: `src/video_text/detector_interface.py`
- Modify: `scripts/run_subtitle_pipeline.py`
- Modify: `requirements.txt`
- Create: `requirements-ppocr-cpu.txt`
- Create: `scripts/setup_ppocr_gpu.sh`
- Test: `tests/video_text/test_ppocrv5_backend.py`
- Test: `tests/video_text/test_cli.py`
- Test: `tests/video_text/test_pipeline.py`

**Interfaces:**
- Consumes: existing `DetectorBackend.detect_batch(images) -> list[tuple[list[Candidate], list[Candidate]]]`.
- Produces: `PPOCRv5MobileBackend(device, batch_size, high_score, low_score, thresh, box_thresh, predictor=None)`, `detect_batch(images)`, `warmup(shape, batch_size=None)`.
- Produces: `PipelineConfig.detector: str` with values `ppocrv5_mobile|fast`.
- Produces: CLI `--detector ppocrv5_mobile|fast`, default `ppocrv5_mobile`.

- [ ] **Step 1: Write backend tests before implementation**

Add tests equivalent to:

```python
class FakeResult(dict):
    pass

class FakePredictor:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []
    def predict(self, images):
        self.calls.append(len(images))
        return self.rows[:len(images)]

def test_ppocr_backend_maps_polygons_to_high_and_low_candidates():
    predictor = FakePredictor([
        FakeResult(
            dt_polys=[
                np.array([[10,10],[110,10],[110,40],[10,40]]),
                np.array([[20,50],[50,50],[50,80],[20,80]]),
            ],
            dt_scores=np.array([.93,.61]),
        ),
    ])
    backend=PPOCRv5MobileBackend(
        device="cpu", batch_size=4,
        high_score=.84, low_score=.50,
        predictor=predictor,
    )
    high,low=backend.detect_batch([np.zeros((100,200,3),np.uint8)])[0]
    assert high[0].bbox.tolist()==[10,10,110,40]
    assert low[0].bbox.tolist()==[20,50,50,80]
    assert high[0].source=="ppocrv5_mobile"
```

Also test that scores below 0.50 are omitted and batch length is preserved.

- [ ] **Step 2: Run backend tests and verify RED**

Run:
`pytest -q tests/video_text/test_ppocrv5_backend.py`

Expected: FAIL because `PPOCRv5MobileBackend` does not exist.

- [ ] **Step 3: Implement `PPOCRv5MobileBackend`**

Implement lazy PaddleOCR import so unit tests do not require Paddle runtime:

```python
class PPOCRv5MobileBackend:
    model_name="PP-OCRv5_mobile_det"

    def __init__(
        self, device="auto", batch_size=16,
        high_score=.84, low_score=.50,
        thresh=.30, box_thresh=.50,
        predictor=None,
    ):
        self.device=resolve_ppocr_device(device)
        self.batch_size=int(batch_size)
        self.high_score=float(high_score)
        self.low_score=float(low_score)
        self.thresh=float(thresh)
        self.box_thresh=float(box_thresh)
        self.precision="fp32"
        if predictor is None:
            from paddleocr import TextDetection
            paddle_device="gpu:0" if self.device.type=="cuda" else "cpu"
            predictor=TextDetection(
                model_name=self.model_name,
                device=paddle_device,
                enable_hpi=False,
                thresh=self.thresh,
                box_thresh=self.box_thresh,
            )
        self.predictor=predictor
```

`detect_batch()` converts each polygon to axis-aligned `[x1,y1,x2,y2]`, creates HIGH/LOW `Candidate` objects, and returns exactly one pair per input image.

- [ ] **Step 4: Add detector selection to config/CLI**

In `PipelineConfig` add:

```python
detector: str="ppocrv5_mobile"
ppocr_thresh: float=.30
ppocr_box_thresh: float=.50
```

Validate detector choice and threshold ranges.

Change `SubtitlePipeline._default_backend()` to instantiate PP-OCR when `detector=="ppocrv5_mobile"`, otherwise retain existing FAST backend.

Add CLI flags:
```text
--detector ppocrv5_mobile|fast
--ppocr-thresh 0.30
--ppocr-box-thresh 0.50
```

The production CLI default becomes PP-OCRv5 Mobile.

- [ ] **Step 5: Add dependency manifests**

Append `paddleocr==3.7.0` to `requirements.txt`.

Create `requirements-ppocr-cpu.txt`:
```text
-r requirements.txt
paddlepaddle==3.3.1
```

Create `scripts/setup_ppocr_gpu.sh` that installs base requirements and:
```bash
python -m pip install paddlepaddle-gpu==3.3.1   -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

Do not commit a model cache.

- [ ] **Step 6: Run targeted tests**

Run:
`pytest -q tests/video_text/test_ppocrv5_backend.py tests/video_text/test_cli.py tests/video_text/test_pipeline.py`

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/video_text/ppocrv5_backend.py src/video_text/detector_interface.py   src/video_text/pipeline.py scripts/run_subtitle_pipeline.py requirements.txt   requirements-ppocr-cpu.txt scripts/setup_ppocr_gpu.sh   tests/video_text/test_ppocrv5_backend.py tests/video_text/test_cli.py   tests/video_text/test_pipeline.py
git commit -m "feat: add PP-OCRv5 mobile detector backend"
```

---

### Task 2: Lightweight PP-OCR temporal hysteresis and transition safety

**Files:**
- Create: `src/video_text/ppocr_temporal.py`
- Modify: `src/video_text/pipeline.py`
- Test: `tests/video_text/test_ppocr_temporal.py`

**Interfaces:**
- Consumes: `list[FrameDetections]`, existing `SubtitleTrack`, `TrackObservation`, `AssociationConfig`, `build_provisional_tracks`.
- Produces: `PPOCRTemporalConfig`.
- Produces: `build_ppocr_temporal_tracks(frames, frame_width, frame_height, config) -> tuple[list[SubtitleTrack], dict]`.
- Produces metrics `ppocr_track_count`, `ppocr_low_backfill_count`, `ppocr_gap_fill_count`, `ppocr_transition_gap_preserved_count`.

- [ ] **Step 1: Write RED tests for hysteresis**

Cover:
1. LOW frame followed by HIGH frames is retained.
2. LOW-only noise does not create a confirmed track.
3. A 1-frame gap inside same stable geometry is filled.
4. A blank frame between geometrically different subtitles is not filled.
5. High-confidence text far outside the dominant subtitle band is filtered.

Use synthetic `FrameDetections` only; no video I/O.

- [ ] **Step 2: Run RED tests**

Run:
`pytest -q tests/video_text/test_ppocr_temporal.py`

Expected: FAIL because temporal module does not exist.

- [ ] **Step 3: Implement high/low association**

Use `AssociationConfig` with:
```python
AssociationConfig(
    iou_gate=.20,
    center_gate=.04,
    min_width_ratio=.65,
    max_width_ratio=1.55,
    min_height_ratio=.65,
    max_height_ratio=1.55,
    max_frame_gap=2,
    backfill_frames=2,
)
```

Require at least one HIGH observation to confirm a normal track.

Learn dominant subtitle vertical band from stable confirmed tracks with >=8 observed frames; use median center-y and a +/-70 px band at 720x1280, scaled by frame height relative to 1280.

- [ ] **Step 4: Split geometry transitions before gap filling**

Implement a helper:
```python
def same_subtitle_geometry(a, b) -> bool:
    # IoU >= .55
    # width ratio <= 1.25
    # height ratio <= 1.25
    # center distance <= scaled 22 px
```

Split tracks whenever geometry fails this predicate.

Fill an internal gap of 1-2 frames only inside a single geometry-consistent segment.

- [ ] **Step 5: Run targeted tests**

Run:
`pytest -q tests/video_text/test_ppocr_temporal.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/video_text/ppocr_temporal.py src/video_text/pipeline.py   tests/video_text/test_ppocr_temporal.py
git commit -m "feat: add lightweight PP-OCR temporal hysteresis"
```

---

### Task 3: Glyph-safe short-line recovery and 57-second regression

**Files:**
- Modify: `src/video_text/ppocr_temporal.py`
- Reuse: `src/video_text/text_enhancement.py`
- Test: `tests/video_text/test_ppocr_temporal.py`
- Create: `tests/video_text/test_ppocr_glyph_geometry.py`

**Interfaces:**
- Consumes: confirmed strong PP-OCR tracks plus unmatched/persistent LOW short-line detections.
- Produces: `measure_short_line_glyph_extent(frame, search_box) -> np.ndarray | None`.
- Produces: `glyph_safe_short_bbox(observed_boxes, glyph_extents, parent_bbox, frame_shape) -> np.ndarray`.
- Produces: `recover_ppocr_short_lines(source, strong_tracks, frames, frame_width, frame_height, config) -> tuple[list[SubtitleTrack], dict]`.

- [ ] **Step 1: Write regression test reproducing the 57s clipping**

Use representative geometry from the measured failure:

```python
def test_short_line_bbox_encloses_visible_glyph_support_without_overlapping_parent():
    observed=[
        [349,891,368,920],
        [354,897,361,910],
        [354,894,361,917],
    ]
    glyph_extents=[
        [340,882,379,924],
        [340,882,379,924],
    ]
    parent=[79,821,639,880]

    box=glyph_safe_short_bbox(
        observed_boxes=observed,
        glyph_extents=glyph_extents,
        parent_bbox=parent,
        frame_shape=(1280,720),
    )

    assert box[0] <= 340
    assert box[1] <= 882
    assert box[2] >= 379
    assert box[3] >= 924
    assert box[1] >= parent[3] + 1
```

Also test that an extreme single-frame glyph outlier does not inflate the box excessively.

- [ ] **Step 2: Run RED geometry tests**

Run:
`pytest -q tests/video_text/test_ppocr_glyph_geometry.py`

Expected: FAIL because helper does not exist.

- [ ] **Step 3: Implement glyph extent measurement**

For a small search crop around the weak track, call existing `white_black_text_mask()`.

Across representative frames:
- collect non-empty mask extents;
- reject isolated spatial outliers;
- compute robust union using the retained extents;
- preserve dark-outline safety with adaptive padding.

Do not OCR/recognize text.

- [ ] **Step 4: Implement adaptive short-line padding**

Use:
```python
pad_x=max(4, round(glyph_height*.10))
pad_y=max(3, round(glyph_height*.07))
```

The canonical box is the union of robust observed detector evidence and robust glyph extents, expanded by adaptive pad and clipped to frame bounds.

For the parent line seam:
```python
child_y1=max(child_y1, int(parent_y2)+1)
```

Only apply that clamp if glyph support still remains inside the child box. If the parent box itself intrudes into measured child glyph support, move the parent bottom edge up to a stable seam instead of clipping the child.

- [ ] **Step 5: Implement short-line persistence**

A LOW short-line can be recovered when:
- >=3 observed frames;
- observed span >=5 frames;
- median center-x is within scaled 70 px of frame center;
- temporal overlap with a confirmed wide parent line is positive;
- weak line begins no more than 12 frames after parent start;
- vertical distance below parent is within scaled 15..90 px.

Fill the canonical child box across the parent subtitle segment only after those conditions pass.

- [ ] **Step 6: Add seam test**

Test two simultaneous canonical tracks whose original boxes overlap by a few pixels. After seam correction:
- positive-area intersection is zero;
- child glyph support remains enclosed;
- per-track bbox is constant across frames.

- [ ] **Step 7: Run targeted tests**

Run:
`pytest -q tests/video_text/test_ppocr_temporal.py tests/video_text/test_ppocr_glyph_geometry.py`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/video_text/ppocr_temporal.py   tests/video_text/test_ppocr_temporal.py   tests/video_text/test_ppocr_glyph_geometry.py
git commit -m "fix: preserve glyph-safe PP-OCR short-line boxes"
```

---

### Task 4: Integrate PP-OCR temporal path into production pipeline and JSON contract

**Files:**
- Modify: `src/video_text/pipeline.py`
- Modify: `src/video_text/benchmark.py`
- Modify: `src/video_text/io.py`
- Test: `tests/video_text/test_pipeline.py`
- Test: `tests/video_text/test_report_metrics.py`
- Test: `tests/video_text/test_json_output.py`

**Interfaces:**
- Consumes: `build_ppocr_temporal_tracks(...)` and short-line recovery from Task 3.
- Produces: production records with sources `PP_HIGH|PP_LOW_BACKFILL|PP_GAP|PP_WEAK_HOLD`.
- Produces: metrics `detector_name`, PP-OCR thresholds, temporal counts, overlap counts, edge-motion stats.
- Produces: complete per-frame JSON including empty frames.

- [ ] **Step 1: Write pipeline RED tests**

Add a fake PP-OCR backend timeline that covers:
- a normal subtitle;
- a LOW-before-HIGH first frame;
- a stable one-frame internal gap;
- a geometry-changing transition with a blank frame;
- a two-line subtitle with short weak line.

Assert the JSON contains every frame and correct source state.

- [ ] **Step 2: Run RED pipeline tests**

Run:
`pytest -q tests/video_text/test_pipeline.py tests/video_text/test_json_output.py tests/video_text/test_report_metrics.py`

Expected: new PP-OCR tests fail before integration.

- [ ] **Step 3: Add `_process_ppocr()` path**

In `SubtitlePipeline.run()` choose:
```python
if self.config.detector=="ppocrv5_mobile":
    state=self._process_ppocr(source,frames,info,target)
else:
    # existing FAST temporal modes
```

The PP-OCR path must not call V5/V55/FAST-specific weak recovery or EasyOCR recognition.

- [ ] **Step 4: Make cache signature detector-complete**

Include:
- detector name;
- PP-OCR model name;
- high/low score;
- `ppocr_thresh`;
- `ppocr_box_thresh`;
- ROI fraction;
- target frame count.

This prevents FAST cache reuse under PP-OCR and prevents threshold A/B runs from sharing stale detections.

- [ ] **Step 5: Preserve complete JSON**

Ensure JSON has exactly `target` frame entries and empty `boxes: []` for frames without subtitle.

Each box includes:
```json
{
  "track_id": 12,
  "line_id": 1,
  "bbox": [338, 881, 383, 927],
  "source": "PP_WEAK_HOLD"
}
```

- [ ] **Step 6: Add report metrics**

Record at minimum:
- detector/model name;
- Paddle/PaddleOCR versions;
- device;
- batch size;
- ROI;
- high/low/thresh/box_thresh;
- detector seconds/FPS;
- detection-loop seconds/FPS;
- temporal seconds;
- render/encode seconds;
- end-to-end seconds/FPS;
- output frame/dropped frame count;
- overlap frame count;
- ghost/recovery counts;
- edge-motion median/P95;
- RAM/VRAM where available.

- [ ] **Step 7: Run targeted suite**

Run:
`pytest -q tests/video_text/test_pipeline.py tests/video_text/test_json_output.py tests/video_text/test_report_metrics.py`

Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/video_text/pipeline.py src/video_text/benchmark.py src/video_text/io.py   tests/video_text/test_pipeline.py tests/video_text/test_json_output.py   tests/video_text/test_report_metrics.py
git commit -m "feat: make PP-OCRv5 mobile the production pipeline"
```

---

### Task 5: H.264 yuv420p render contract and focused visual verification

**Files:**
- Modify: `src/video_text/io.py`
- Modify: `src/video_text/pipeline.py`
- Test: `tests/video_text/test_io.py`
- Test: `tests/video_text/test_pipeline.py`

**Interfaces:**
- Consumes: production PP-OCR records.
- Produces: H.264 High/yuv420p MP4 with source audio.
- Produces: review frames/clip around configured frame/time range for diagnosis.

- [ ] **Step 1: Write output-format test**

For a tiny fixture, verify:
- output is readable;
- output resolution/FPS/frame count equal source;
- H.264/yuv420p is requested in ffmpeg command;
- source audio mapping is retained.

Mock ffmpeg command construction where codec inspection is unavailable in CI.

- [ ] **Step 2: Run RED test**

Run:
`pytest -q tests/video_text/test_io.py tests/video_text/test_pipeline.py`

Expected: format assertion fails before render command update.

- [ ] **Step 3: Render through ffmpeg directly to production format**

Encode final video with:
```text
-c:v libx264
-profile:v high
-pix_fmt yuv420p
-preset veryfast
-crf 18
-movflags +faststart
-map source-audio when present
-c:a copy
-shortest
```

Avoid the previous yuv444p intermediate result.

- [ ] **Step 4: Run targeted tests**

Run:
`pytest -q tests/video_text/test_io.py tests/video_text/test_pipeline.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/video_text/io.py src/video_text/pipeline.py   tests/video_text/test_io.py tests/video_text/test_pipeline.py
git commit -m "feat: emit playback-safe H264 yuv420p output"
```

---

### Task 6: RTX 3070 full-video verification and benchmark evidence

**Files:**
- Modify: `scripts/benchmark_runtime.py`
- Modify: `README.md`
- Test: `tests/video_text/test_benchmark_runtime.py`

**Interfaces:**
- Consumes: committed PP-OCR production CLI.
- Produces: full boxed MP4, complete JSON, benchmark JSON/log.
- Produces exact command/hash/metrics for report.

- [ ] **Step 1: Update benchmark script for detector-aware runs**

Add `--detector` and PP-OCR threshold arguments.

Warm up PP-OCR through backend `warmup()`.

Always remove the variant detection cache before a timed benchmark.

- [ ] **Step 2: Run benchmark-script tests**

Run:
`pytest -q tests/video_text/test_benchmark_runtime.py`

Expected: PASS after detector-aware implementation.

- [ ] **Step 3: Run full test suite before performance claim**

Run:
`pytest -q`

Expected: all tests pass.

- [ ] **Step 4: Run full 3733-frame RTX 3070 production command**

Use:
```bash
python scripts/run_subtitle_pipeline.py   "/home/plab/Desktop/Nguyen_Anh_Quyet_PLAB/AI/Test/AI Engineer test.mp4"   --output outputs/PP-OCRv5_mobile_PRODUCTION.mp4   --detector ppocrv5_mobile   --device cuda   --batch-size 16   --roi-bottom-fraction .45   --ppocr-thresh .30   --ppocr-box-thresh .50   --high-score .84   --low-score .50   --export-coordinates outputs/PP-OCRv5_mobile_PRODUCTION.json
```

Do not use a pre-existing detection cache.

- [ ] **Step 5: Verify final artifact**

Verify:
- 720x1280;
- 30 FPS;
- 3733 frames;
- H.264 High;
- yuv420p;
- AAC source audio present;
- zero dropped frame;
- zero multiline overlap;
- no ghost transition at frame 1105;
- line 2 around frame 1710 fully encloses visible glyph/outline;
- frame 2552 is recovered;
- stable-track edge-motion median/P95 = 0 px.

Export review frames around frame 975, 1105, 1710, 1750, 2552.

- [ ] **Step 6: Record exact benchmark**

Record:
- GPU/CPU/RAM/VRAM/OS;
- Python, PaddlePaddle, PaddleOCR, OpenCV versions;
- Git commit;
- exact command;
- model and thresholds;
- detector seconds/FPS;
- temporal seconds;
- render seconds;
- end-to-end seconds/FPS;
- RTF;
- output hash and JSON hash.

- [ ] **Step 7: Update README**

Document PP-OCR production setup for GPU and CPU, CLI example, FAST fallback command, and measured RTX 3070 result clearly labeled MEASURED.

- [ ] **Step 8: Commit benchmark docs/source changes**

```bash
git add scripts/benchmark_runtime.py tests/video_text/test_benchmark_runtime.py README.md
git commit -m "docs: record PP-OCRv5 production benchmark"
```

Generated MP4/JSON/log remain untracked.

---

### Task 7: Push exact tested commit and Quyt CPU validation

**Files:**
- No generated runtime artifact committed.
- Optional README follow-up only if CPU setup instructions need correction.

**Interfaces:**
- Consumes: exact tested feature-branch HEAD from Task 6.
- Produces: pushed Git branch/commit and measured Quyt CPU benchmark.

- [ ] **Step 1: Verify clean branch**

Run:
`git status --short`

Expected: empty, excluding ignored model caches/venvs/generated outputs.

Run:
`pytest -q`

Expected: all tests pass immediately before push.

- [ ] **Step 2: Push feature branch**

Push:
```bash
git push -u origin feature/fast-v1-acceleration
```

Record exact:
```bash
git rev-parse HEAD
git remote get-url origin
```

- [ ] **Step 3: Clone/pull exact commit on Quyt**

Use a fresh checkout or reset the benchmark checkout to the exact pushed commit.

Install CPU runtime from:
`requirements-ppocr-cpu.txt`.

- [ ] **Step 4: Run CPU batch sweep, then full CPU validation**

First benchmark a representative 300-frame segment with explicit CPU batch sizes 1, 2, 4, and 8 using the same ROI and thresholds. Select the batch size with the highest measured end-to-end FPS among runs that preserve the GPU production frame coverage.

For every candidate use:
```text
--detector ppocrv5_mobile
--device cpu
--ppocr-thresh .30
--ppocr-box-thresh .50
--high-score .84
--low-score .50
--max-frames 300
```

Then run the full video once with the selected batch size and the same settings, removing `--max-frames 300`.

Measure actual CPU model, RAM, OS, processing seconds/FPS, and output quality.

- [ ] **Step 5: Compare CPU/GPU geometry**

Compare line-count/frame coverage and canonical boxes; CPU output must not introduce subtitle misses, ghost frames, or overlap relative to GPU production output.

- [ ] **Step 6: Report final handoff**

Report:
- Git repository URL;
- branch;
- exact commit;
- RTX 3070 measured benchmark;
- Quyt CPU measured benchmark;
- full-video artifact paths;
- any remaining deferred minor issues.

Do not merge the feature branch unless explicitly requested.
