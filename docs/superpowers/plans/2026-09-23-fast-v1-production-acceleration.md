# FAST V1 Production Acceleration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a detection-only FAST V1 pipeline that preserves the current tight/stable line boxes, runs efficiently on RTX 3070 and CPU, exports review MP4 + per-frame bbox JSON, and emits reproducible benchmark data for the hiring report.

**Architecture:** Keep FAST-B736 and the V1 temporal pipeline as the quality baseline. Add an explicit runtime layer for device/precision/thread configuration, make V1 tightening work without OCR recognition, instrument the pipeline and output contract, then benchmark detector and end-to-end throughput on the RTX 3070 before pushing the exact Git commit for CPU validation on Quyt.

**Tech Stack:** Python 3.10/3.11, PyTorch 2.6, CUDA 12.4, OpenCV, NumPy, FFmpeg, FAST-B736, pytest.

**Spec:** `docs/superpowers/specs/2026-09-23-fast-v1-production-acceleration-design.md`

## Global Constraints

- Primary detector remains FAST-B736 with `fast_base_ic15_736_finetune_ic17mlt.pth`.
- Production mode performs no OCR recognition.
- ROI is configurable from 25%-45% of frame height and defaults to 45%.
- H.264/H.265 MP4 input at common 720p-1080p resolutions must remain supported.
- H.264 output is default; H.265 remains optional.
- Source FPS, dimensions, frame ordering, and audio must be preserved.
- Final output is line-level bbox geometry; two subtitle rows remain separate.
- Static subtitle geometry must not visibly jitter or flicker.
- One/two-frame internal gaps may be reconstructed; true subtitle endings must not create ghost boxes.
- GPU/CPU optimizations are accepted only if visual/coordinate regression is not worse than the V1 golden reference.
- Git repo URL + branch + exact commit hash are the source-of-truth for code; generated benchmark artifacts are report evidence, not source deliverables.

## Review Focus

1. **CUDA exists but `--device cpu` is selected:** no CUDA synchronization, allocation, or memory-stat call may be executed; add a CPU-with-CUDA-present unit test in Task 1/2.
2. **FP16 changes detector logits near threshold:** HIGH/LOW classification and final box geometry must remain within regression tolerance; add FP32-vs-FP16 parity tests in Task 2 and full-video coordinate comparison in Task 6.
3. **OCR disabled removes old V1 content guards:** pixel-only tightening/merge logic must not clip CJK strokes or incorrectly merge different subtitles; add synthetic/noise/transition tests in Task 3.
4. **Video has frames with no subtitles:** JSON must still represent every processed frame with an empty `boxes` list and renderer must emit every source frame; add output-contract tests in Task 4.
5. **Audio is shorter/longer than video or absent:** mux must never truncate the video and must tolerate no-audio input; extend IO tests in Task 4.

---

## File Structure

### New files
- `src/video_text/runtime.py` — device/precision resolution, CPU thread tuning, CUDA-safe synchronization and memory helpers.
- `src/video_text/benchmark.py` — environment provenance, phase timing, percentile/RTF calculations, benchmark JSON serialization.
- `scripts/benchmark_runtime.py` — detector/full-pipeline benchmark runner and batch/precision sweep.
- `scripts/compare_coordinates.py` — compare CPU/GPU/FP32/FP16 coordinate JSON and quality invariants.
- `tests/video_text/test_runtime.py` — runtime/device selection tests.
- `tests/video_text/test_benchmark.py` — metric/provenance/percentile tests.
- `tests/video_text/test_coordinate_compare.py` — coordinate parity checker tests.

### Modified files
- `scripts/run_subtitle_pipeline.py` — production CLI options and detection-only defaults.
- `src/video_text/fast_backend.py` — device, precision, batch size, pinned/non-blocking transfer, CUDA autocast/warm-up.
- `src/video_text/pipeline.py` — runtime wiring, OCR-free V1 path, phase timings, metrics, render options.
- `src/video_text/v1_processor.py` — conservative pixel-only V1 glyph tightening and visual-only same-content stitch fallback.
- `src/video_text/io.py` — per-frame bbox JSON contract and robust output/mux metadata.
- `src/video_text/display_shape.py` — configurable line thickness while keeping anti-aliased unfilled boxes.
- `tests/video_text/test_cli.py` — CLI device/precision/detection-only tests.
- `tests/video_text/test_fast_backend_levels.py` — backend runtime/parity tests.
- `tests/video_text/test_v1_geometry.py` / `test_v1_processor.py` — OCR-free tight-box safety tests.
- `tests/video_text/test_pipeline.py` / `test_v1_pipeline_integration.py` — end-to-end detection-only tests.
- `tests/video_text/test_video_io.py` — full-frame JSON and mux tests.
- `README.md` / `docs/REPRODUCIBILITY.md` — exact production and benchmark commands.

---

### Task 1: Add explicit runtime/device contract and make detection-only the production default

**Files:**
- Create: `src/video_text/runtime.py`
- Modify: `scripts/run_subtitle_pipeline.py`
- Modify: `src/video_text/pipeline.py`
- Test: `tests/video_text/test_runtime.py`
- Test: `tests/video_text/test_cli.py`

**Interfaces:**
- Produces: `resolve_device(requested: str) -> torch.device`
- Produces: `resolve_precision(requested: str, device: torch.device) -> str`
- Produces: `configure_runtime(device: torch.device, cpu_threads: int | None) -> dict`
- Produces: `synchronize(device: torch.device) -> None`
- `PipelineConfig` gains `device`, `precision`, `batch_size`, `cpu_threads`, `box_thickness`; `validate_chinese` defaults to `False`.

- [ ] **Step 1: Write failing runtime tests**

```python
# tests/video_text/test_runtime.py
import torch
import pytest
from src.video_text.runtime import resolve_device, resolve_precision


def test_auto_uses_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_explicit_cpu_wins_even_when_cuda_exists(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cpu").type == "cpu"


def test_explicit_cuda_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_fp16_is_cuda_only():
    with pytest.raises(ValueError, match="fp16"):
        resolve_precision("fp16", torch.device("cpu"))
```

- [ ] **Step 2: Run targeted tests and verify failure**

Run: `pytest -q tests/video_text/test_runtime.py`
Expected: FAIL because `src.video_text.runtime` does not exist.

- [ ] **Step 3: Implement minimal runtime helpers**

```python
# src/video_text/runtime.py
from __future__ import annotations
import os
import torch


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be auto, cuda, or cpu")


def resolve_precision(requested: str, device: torch.device) -> str:
    requested = str(requested).lower()
    if requested == "auto":
        return "fp16" if device.type == "cuda" else "fp32"
    if requested not in {"fp32", "fp16"}:
        raise ValueError("precision must be auto, fp32, or fp16")
    if requested == "fp16" and device.type != "cuda":
        raise ValueError("fp16 is supported only on CUDA")
    return requested


def configure_runtime(device: torch.device, cpu_threads: int | None = None) -> dict:
    if device.type == "cpu":
        threads = int(cpu_threads or max(1, (os.cpu_count() or 1) // 2))
        torch.set_num_threads(threads)
        return {"cpu_threads": threads}
    torch.backends.cudnn.benchmark = True
    return {"cpu_threads": 0}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
```

- [ ] **Step 4: Extend CLI tests for production defaults**

Add parser-level/subprocess assertions that:
- `--device` accepts `auto|cpu|cuda`;
- `--precision` accepts `auto|fp32|fp16`;
- `--batch-size` is positive;
- `--cpu-threads` is optional positive int;
- `--box-thickness` is positive;
- `validate_chinese` defaults to `False`;
- `--validate-chinese` remains an explicit diagnostic opt-in.

- [ ] **Step 5: Wire CLI values into `PipelineConfig`**

Add:

```python
p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
p.add_argument("--precision", choices=["auto", "fp32", "fp16"], default="auto")
p.add_argument("--batch-size", type=int, default=16)
p.add_argument("--cpu-threads", type=int, default=0)
p.add_argument("--box-thickness", type=int, default=2)
p.add_argument("--validate-chinese", dest="validate_chinese", action="store_true", default=False)
p.add_argument("--no-validate-chinese", dest="validate_chinese", action="store_false")
```

Reject non-positive batch size/thickness and negative CPU thread count in `PipelineConfig.__post_init__`.

- [ ] **Step 6: Run Task 1 tests**

Run: `pytest -q tests/video_text/test_runtime.py tests/video_text/test_cli.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/video_text/runtime.py scripts/run_subtitle_pipeline.py src/video_text/pipeline.py tests/video_text/test_runtime.py tests/video_text/test_cli.py
git commit -m "feat: add explicit cpu cuda runtime configuration"
```

---

### Task 2: Accelerate FAST backend on CUDA while preserving CPU behavior

**Files:**
- Modify: `src/video_text/fast_backend.py`
- Modify: `src/video_text/pipeline.py`
- Test: `tests/video_text/test_fast_backend_levels.py`
- Test: `tests/video_text/test_pipeline.py`

**Interfaces:**
- `FastBackend(..., device: str|torch.device, precision: str, batch_size: int, pin_memory: bool=True)`
- Produces attributes: `device: torch.device`, `precision: str`, `batch_size: int`
- Produces: `warmup(image_shape: tuple[int,int,int], batch_size: int|None=None) -> None`

- [ ] **Step 1: Add failing backend configuration tests**

```python
def test_fake_backend_cpu_path_never_calls_cuda_sync(monkeypatch):
    called = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a, **k: called.append(True))
    backend = FakeBackend()
    backend.device = torch.device("cpu")
    backend.batch_size = 2
    # pipeline detection on CPU must not consult global CUDA availability
    # assertion is completed in test_pipeline with a synthetic video.
    assert called == []
```

Add a test that `FastBackend`/a lightweight subclass stores `precision` and `batch_size`, and that FP32/FP16 decode produces the same HIGH/LOW counts for fixed synthetic logits.

- [ ] **Step 2: Run targeted tests and verify failure**

Run: `pytest -q tests/video_text/test_fast_backend_levels.py tests/video_text/test_pipeline.py`
Expected: FAIL on missing runtime attributes/device-safe sync behavior.

- [ ] **Step 3: Implement device-aware initialization**

In `FastBackend.__init__`:

```python
self.device = torch.device(device)
self.precision = str(precision)
self.batch_size = int(batch_size)
self.pin_memory = bool(pin_memory and self.device.type == "cuda")
self.model = model.to(self.device).eval()
```

Keep weights FP32; use autocast for CUDA FP16 rather than permanently converting the model so FP32 and FP16 share one checkpoint path.

- [ ] **Step 4: Optimize host-to-device transfer**

Replace unconditional `.to(self.device)` with:

```python
host = torch.from_numpy(batch)
if self.pin_memory:
    host = host.pin_memory()
x = host.to(self.device, non_blocking=self.pin_memory)
```

Keep preprocessing numerically identical to the golden V1 path.

- [ ] **Step 5: Add CUDA autocast only around neural forward**

```python
def _forward_model(self, x):
    self.forward_calls += 1
    enabled = self.device.type == "cuda" and self.precision == "fp16"
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled):
        f = self.model.backbone(x)
        f = self.model.neck(f)
        out = self.model.det_head(f)
    return out
```

Convert score tensors to FP32 before NumPy transfer in `_decode_scored_components` so CPU postprocessing behavior remains stable.

- [ ] **Step 6: Replace global CUDA checks in pipeline timing**

In `SubtitlePipeline._detect`, call `synchronize(self.backend.device)` before/after timed detector calls only when the backend device is CUDA. Never use `torch.cuda.is_available()` to decide whether a CPU backend synchronizes.

- [ ] **Step 7: Add a warm-up method**

`warmup()` creates one representative zero ROI batch and performs 3 untimed forwards; call it once after backend creation for benchmark runs, not for cache hits or `--dry-probe`.

- [ ] **Step 8: Run backend/pipeline tests**

Run: `pytest -q tests/video_text/test_fast_backend_levels.py tests/video_text/test_pipeline.py`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/video_text/fast_backend.py src/video_text/pipeline.py tests/video_text/test_fast_backend_levels.py tests/video_text/test_pipeline.py
git commit -m "perf: accelerate FAST inference across cpu and cuda"
```

---

### Task 3: Preserve V1 tight-box quality with OCR completely disabled

**Files:**
- Modify: `src/video_text/v1_processor.py`
- Modify: `src/video_text/pipeline.py`
- Test: `tests/video_text/test_v1_processor.py`
- Test: `tests/video_text/test_v1_geometry.py`
- Test: `tests/video_text/test_v1_pipeline_integration.py`

**Interfaces:**
- `tighten_v1_static_tracks_with_temporal_glyphs(..., recognizer=None, ...)` must remain useful without a recognizer.
- `merge_v1_same_content_tracks(..., recognizer=None, ...)` gains a conservative visual-only fallback.
- Production V1 must never construct `EasyOCRChineseRecognizer` unless `validate_chinese=True` is explicitly requested.

- [ ] **Step 1: Add failing OCR-free tightening test**

Create a static synthetic subtitle with a deliberately loose anchor and persistent outlined glyphs. Call `tighten_v1_static_tracks_with_temporal_glyphs(..., recognizer=None)` and assert:
- output area is at least 2% smaller than anchor;
- bbox still contains all known glyph extents plus configured safety pad;
- every frame in the track receives exactly the same box.

Representative assertion:

```python
assert new_area < 0.98 * old_area
assert box[0] <= glyph_x1 - 2
assert box[2] >= glyph_x2 + 2
assert len({tuple(obs.bbox) for obs in track.observations.values()}) == 1
```

- [ ] **Step 2: Add failing one-frame-noise and thin-glyph tests**

Use a 9-15 frame sequence where one frame contains a bright edge outside the subtitle and another case contains a persistent thin horizontal glyph. OCR-free tightening must ignore the transient edge and retain the thin glyph.

- [ ] **Step 3: Implement conservative pixel-only guard**

When `recognizer is None`, `_guard_temporal_candidate_edges` must no longer return the original anchor unconditionally. Accept the multi-frame `_temporal_glyph_candidate` only when all are true:
- candidate has positive area;
- candidate is inside the trusted temporal anchor (no unsupported expansion);
- candidate width >= 70% of anchor width;
- candidate height >= 60% of anchor height;
- candidate area is between 45% and 98% of anchor area;
- the configured safety pad is already included by `_temporal_glyph_candidate`.

Otherwise return the anchor unchanged. This makes OCR-free behavior conservative rather than disabling tightening.

- [ ] **Step 4: Always run temporal-glyph tightening in production V1**

Replace the current `if recognizer is not None else no-op metrics` branch with an unconditional call to `tighten_v1_static_tracks_with_temporal_glyphs(..., recognizer=recognizer)`. Keep `tighten_v1_static_tracks_with_ocr` out of the production detection-only path; when recognition is disabled, report `ocr_tightened_track_count=0` and ratio `1.0` without invoking OCR code.

- [ ] **Step 5: Add visual-only same-content stitch test**

Construct adjacent single-line tracks separated by <=1 frame with identical/similar boundary crops and no recognizer. Assert conservative merge occurs only above a high visual threshold; add a second case with different visual content and assert no merge.

- [ ] **Step 6: Implement conservative visual-only merge fallback**

In `merge_v1_same_content_tracks`, retain existing OCR-assisted rules when text evidence exists. When both sides have no text evidence, allow merge only when:
- `gap <= 1`;
- boundary visual similarity >= 0.94;
- both tracks are single-line eligible tracks;
- median width/height ratios stay within 0.85..1.18.

This restores obvious fragmentation stitching without reading text.

- [ ] **Step 7: Prove recognizer is not initialized in production mode**

Add an integration test that monkeypatches `EasyOCRChineseRecognizer` to raise if constructed, runs `PipelineConfig(temporal_mode="v1", validate_chinese=False)`, and asserts the pipeline succeeds.

- [ ] **Step 8: Run V1 quality tests**

Run: `pytest -q tests/video_text/test_v1_processor.py tests/video_text/test_v1_geometry.py tests/video_text/test_v1_pipeline_integration.py`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/video_text/v1_processor.py src/video_text/pipeline.py tests/video_text/test_v1_processor.py tests/video_text/test_v1_geometry.py tests/video_text/test_v1_pipeline_integration.py
git commit -m "perf: preserve tight V1 boxes without OCR recognition"
```

---

### Task 4: Add report-grade metrics and complete per-frame JSON/output contract

**Files:**
- Create: `src/video_text/benchmark.py`
- Modify: `src/video_text/pipeline.py`
- Modify: `src/video_text/io.py`
- Modify: `src/video_text/display_shape.py`
- Test: `tests/video_text/test_benchmark.py`
- Test: `tests/video_text/test_video_io.py`
- Test: `tests/video_text/test_pipeline.py`

**Interfaces:**
- Produces: `collect_environment_metadata(project_root: Path, checkpoint: Path) -> dict`
- Produces: `latency_summary(samples_ms: list[float]) -> dict`
- Produces: `real_time_factor(elapsed_seconds: float, duration_seconds: float) -> float`
- `write_coordinate_json(..., frame_count: int, fps: float, ...)` adds top-level `frames` while retaining flat `records` for backward compatibility.

- [ ] **Step 1: Write failing benchmark helper tests**

```python
def test_latency_summary_reports_mean_p50_p95():
    s = latency_summary([1.0, 2.0, 3.0, 100.0])
    assert s["mean_ms"] == 26.5
    assert 2.0 <= s["p50_ms"] <= 3.0
    assert s["p95_ms"] > 3.0


def test_real_time_factor():
    assert real_time_factor(50.0, 100.0) == 0.5
```

- [ ] **Step 2: Add per-frame JSON failing test**

For 3 processed frames with a box only on frame 1, assert JSON contains exactly 3 entries in `frames`, with `frames[0]["boxes"] == []`, one box on frame 1, and `frames[2]["boxes"] == []`.

- [ ] **Step 3: Implement environment provenance**

Collect without network access:
- OS/platform;
- Python version;
- torch/OpenCV versions;
- CUDA version and CUDA device name when used;
- CPU logical count;
- Git commit via `git rev-parse HEAD` when available;
- FAST commit via `git -C model/FAST rev-parse HEAD`;
- checkpoint name + SHA-256.

Failures to obtain optional provenance must yield `None`, not abort video processing.

- [ ] **Step 4: Instrument pipeline phases**

Record at minimum:
- decode/detection loop elapsed;
- detector elapsed per batch and detector FPS;
- temporal/postprocess elapsed;
- render/transcode/mux elapsed;
- total elapsed/end-to-end FPS;
- per-frame detector latency mean/P50/P95;
- real-time factor;
- output frame count and dropped-frame count;
- source duration.

Only synchronize CUDA around detector timing when the selected device is CUDA.

- [ ] **Step 5: Make VRAM metrics device-safe**

Reset/read CUDA peak memory only when `backend.device.type == "cuda"`; CPU runs report `peak_vram_mb=0.0`. Add `device`, `precision`, `batch_size`, and `cpu_threads` to metrics.

- [ ] **Step 6: Add complete `frames` JSON while preserving existing `records`**

Build:

```json
{
  "frame": 79,
  "timestamp": 2.6333333333,
  "boxes": [
    {"track_id": 6, "subtitle_id": 6, "line_id": 0,
     "bbox": [83,824,637,876], "reconstructed": false}
  ]
}
```

for every processed frame 0..N-1. Keep existing `records`, `events`, and `display_shapes` so regression tooling remains compatible.

- [ ] **Step 7: Keep output boxes visually clean**

Use existing `cv2.LINE_AA`, no fill, and configurable `box_thickness` (default 2). Add a test that `draw_line_rectangles` does not modify interior pixels away from the border on a blank frame.

- [ ] **Step 8: Extend mux tests**

Add no-audio input and source-audio-longer-than-video cases. In all cases output frame count must equal video-only input frame count.

- [ ] **Step 9: Run report/output tests**

Run: `pytest -q tests/video_text/test_benchmark.py tests/video_text/test_video_io.py tests/video_text/test_pipeline.py`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add src/video_text/benchmark.py src/video_text/pipeline.py src/video_text/io.py src/video_text/display_shape.py tests/video_text/test_benchmark.py tests/video_text/test_video_io.py tests/video_text/test_pipeline.py
git commit -m "feat: emit report-grade metrics and complete bbox json"
```

---

### Task 5: Build benchmark sweep and coordinate-regression tooling

**Files:**
- Create: `scripts/benchmark_runtime.py`
- Create: `scripts/compare_coordinates.py`
- Create: `tests/video_text/test_coordinate_compare.py`
- Modify: `README.md`
- Modify: `docs/REPRODUCIBILITY.md`

**Interfaces:**
- `benchmark_runtime.py INPUT --device cuda --batches 4,8,16,32 --precisions fp32,fp16 --max-frames 600 --output-dir ...`
- `compare_coordinates.py GOLDEN.json CANDIDATE.json --max-edge-delta 2 --max-missing-frame-rate 0.0`

- [ ] **Step 1: Write failing coordinate comparison tests**

Synthetic JSON fixtures must verify:
- identical files pass;
- 1 px edge differences pass under 2 px tolerance;
- missing subtitle frame fails;
- added ghost frame fails;
- multiline overlap increase fails when metric metadata worsens.

- [ ] **Step 2: Implement coordinate comparator**

Compare by `(frame, subtitle_id, line_id)` with metrics:
- matched box count;
- missing key count;
- extra key count;
- mean/P95/max absolute edge delta;
- boxed-frame set equality;
- ghost/missing frame lists;
- key quality metrics (`multiline_overlap_frame_count`, stabilized motion, bbox excess area) when present.

Exit non-zero when configured tolerances fail.

- [ ] **Step 3: Implement benchmark sweep**

For each variant:
1. run short warm-up;
2. run detector/full pipeline on the requested frame limit;
3. write one JSON row with command/config/provenance/metrics;
4. continue after OOM by recording failure and clearing CUDA cache;
5. never silently reduce batch size.

Default RTX 3070 sweep: batches `4,8,16,32`, precisions `fp32,fp16`, first 600 frames.

- [ ] **Step 4: Add selection rule**

Rank only variants whose coordinate comparator passes against FP32 golden output. Select highest end-to-end FPS; on ties within 3%, choose the smaller batch / FP32 variant for lower memory and safer reproducibility.

- [ ] **Step 5: Document exact commands**

README server command:

```bash
python scripts/benchmark_runtime.py "AI Engineer test.mp4" \
  --device cuda --batches 4,8,16,32 --precisions fp32,fp16 \
  --max-frames 600 --output-dir outputs/bench_rtx3070
```

Final run command is written only after sweep selection; README must contain the exact selected measured configuration before final commit.

CPU Quyt command:

```powershell
python scripts/run_subtitle_pipeline.py "AI Engineer test.mp4" `
  --output outputs\FAST_V1_CPU_Quyt.mp4 `
  --device cpu --precision fp32 --batch-size 1 `
  --no-validate-chinese

This command is the CPU golden baseline only. After the CPU sweep, copy the exact selected command emitted by `benchmark_runtime.py` into `docs/REPRODUCIBILITY.md`.
```

- [ ] **Step 6: Run tooling tests**

Run: `pytest -q tests/video_text/test_coordinate_compare.py`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/benchmark_runtime.py scripts/compare_coordinates.py tests/video_text/test_coordinate_compare.py README.md docs/REPRODUCIBILITY.md
git commit -m "feat: add reproducible runtime benchmark tooling"
```

---

### Task 6: Verify the complete branch before any performance claim

**Files:**
- No new production files unless failures require fixes.

**Interfaces:**
- Consumes all prior tasks.
- Produces a clean test baseline before real hardware benchmarking.

- [ ] **Step 1: Run the complete test suite**

Run: `pytest -q`
Expected: all tests PASS; no unexpected skips other than private-video-dependent tests when the fixture is absent.

- [ ] **Step 2: Run repository artifact verifier**

Run: `python scripts/verify_consistent_final.py`
Expected: reference artifact contract PASS.

- [ ] **Step 3: Run a 300-frame detection-only smoke on server CUDA FP32**

```bash
python scripts/run_subtitle_pipeline.py "AI Engineer test.mp4" \
  --output outputs/smoke_cuda_fp32.mp4 \
  --device cuda --precision fp32 --batch-size 16 \
  --temporal-mode v1 --no-validate-chinese --max-frames 300
```

Expected: MP4 playable, JSON generated, frame count 300, OCR recognizer never initialized.

- [ ] **Step 4: Inspect smoke metrics**

Assert/report:
- dropped frames = 0;
- ghost-box regression checks pass;
- multiline overlap frame count = 0;
- stabilized edge motion median/P95 do not regress against V1 thresholds;
- audio mux does not truncate output;
- JSON has exactly 300 per-frame entries.

- [ ] **Step 5: Commit any verification-only fixes separately**

If fixes were needed, rerun Steps 1-4 and commit with a focused `fix:` message. Otherwise do not create an empty commit.

---

### Task 7: Benchmark RTX 3070, select fastest passing configuration, and generate final server artifacts

**Files:**
- Generated only: `outputs/bench_rtx3070/*.json`, selected final MP4/JSON/log (do not Git-add large generated artifacts).
- Modify only after measured selection: `README.md`, `docs/REPRODUCIBILITY.md`.

**Interfaces:**
- Produces measured RTX 3070 rows for the report and one selected final GPU MP4/JSON.

- [ ] **Step 1: Record server hardware/software baseline**

Capture CPU, GPU, VRAM, RAM, OS, Python, torch, CUDA, OpenCV, FAST commit, checkpoint SHA-256, and current project Git commit into benchmark metadata.

- [ ] **Step 2: Establish a 600-frame FP32 golden run**

Use batch 16 initially. Save MP4/JSON as the geometry reference for the sweep.

- [ ] **Step 3: Sweep batch/precision candidates**

Run detector/full pipeline variants for batch `4,8,16,32` and `fp32/fp16`, recording OOM explicitly rather than hiding it.

- [ ] **Step 4: Compare every candidate against FP32 golden**

Run `scripts/compare_coordinates.py`. Reject any variant with missing/extra subtitle frames, new overlap, or edge delta beyond tolerance.

- [ ] **Step 5: Select the fastest passing variant**

Choose by measured end-to-end FPS, with the 3% tie rule from Task 5. Record selected batch, precision, peak VRAM, detector FPS, end-to-end FPS, P50/P95 latency, total processing time, and RTF.

- [ ] **Step 6: Run the complete 3733-frame supplied video**

Use the selected configuration with `--max-frames 0`. Produce:
- `outputs/FAST_V1_RTX3070_FINAL.mp4`
- `outputs/FAST_V1_RTX3070_FINAL.json`
- benchmark metrics/log JSON.

- [ ] **Step 7: Verify final GPU artifact visually and numerically**

Run full coordinate comparison against the prior V1 reference where comparable. Generate/check contact sheets around regressions or track transitions. Confirm the final MP4 is playable and audio is present when source audio exists.

- [ ] **Step 8: Update reproducibility docs with measured values**

Update README/docs with the exact selected command and measured RTX 3070 configuration.

- [ ] **Step 9: Run full tests once more**

Run: `pytest -q`
Expected: PASS.

- [ ] **Step 10: Commit measured configuration docs**

```bash
git add README.md docs/REPRODUCIBILITY.md
git commit -m "docs: record RTX 3070 production benchmark"
```

---

### Task 8: Push exact tested Git commit and validate CPU on Quyt

**Files:**
- Git source only; generated Quyt benchmark/video/JSON remain untracked report evidence.

**Interfaces:**
- Produces Git repo URL/branch/commit for reproducibility plus measured Quyt CPU data.

- [ ] **Step 1: Verify server worktree is clean except intentionally ignored outputs**

Run: `git status --short`
Expected: no uncommitted production/test/doc changes.

- [ ] **Step 2: Push the tested branch**

```bash
git push origin main
```

Record: `git rev-parse HEAD` and `git remote get-url origin`.

- [ ] **Step 3: Clone/pull exactly that commit on Quyt**

Do not copy source files manually. Verify `git rev-parse HEAD` equals the server commit before benchmarking.

- [ ] **Step 4: Run Quyt CPU batch sweep**

Test practical CPU batches `1,2,4,8` on a fixed 600-frame sample using FP32 only. Test a small set of thread counts around physical-core count (for example 8, 12, 16 when appropriate) and record every configuration explicitly.

- [ ] **Step 5: Select fastest passing CPU configuration**

Coordinate-compare each candidate against server FP32 golden. Reject geometry regressions; choose highest end-to-end FPS among passing runs.

- [ ] **Step 6: Run complete Quyt CPU video**

Produce:
- `outputs/FAST_V1_Quyt_CPU_FINAL.mp4`
- `outputs/FAST_V1_Quyt_CPU_FINAL.json`
- benchmark metrics/log JSON.

- [ ] **Step 7: Record report rows**

Record measured CPU model, RAM, OS, Python/torch/OpenCV versions, batch, thread count, total time, detector FPS, end-to-end FPS, mean/P50/P95 latency, RTF, peak RAM, frame-drop count, and quality invariants.

- [ ] **Step 8: Final parity check**

Run `scripts/compare_coordinates.py` between selected server GPU JSON and Quyt CPU JSON. CPU/GPU outputs must satisfy the coordinate tolerance and have identical boxed-frame sets unless a documented floating-point threshold difference is manually reviewed and accepted.

---

## Final Verification Gate

Before calling the implementation complete:

```bash
pytest -q
python scripts/verify_consistent_final.py
```

Then verify the selected full-video artifacts:
- RTX 3070 final MP4 opens and plays with expected audio.
- RTX 3070 JSON contains all 3733 frame entries.
- Quyt CPU final MP4 opens and plays.
- Quyt CPU JSON contains all 3733 frame entries.
- No dropped frames.
- No multiline overlap regression.
- No new ghost frames at subtitle endings.
- Static V1 edge-motion remains effectively locked.
- Tight-box metric is not worse than the golden V1 output.
- Git repo/branch/commit in benchmark metadata exactly match the pushed source.

