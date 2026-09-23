# PP-OCRv5 Mobile Temporal Subtitle Detector

Production-oriented detector for hard-coded Chinese subtitles in MP4 video.

The default production path uses **PP-OCRv5 Mobile Det** on the bottom subtitle
ROI, followed by lightweight temporal hysteresis and stable per-line geometry.
FAST-B remains available as an explicit fallback/reference backend.

## Production behavior

- bottom ROI detection only (default: bottom 45%);
- PP-OCRv5 Mobile text detection, no OCR recognition on the critical path;
- HIGH/LOW score hysteresis for temporal continuity;
- stable canonical bbox per subtitle line;
- recovery of short/single-character second lines;
- glyph-safe adaptive padding so boxes do not clip visible strokes/outlines;
- stable multiline seam with zero positive-area overlap;
- transition protection to avoid 1-2 frame ghost boxes;
- complete per-frame coordinate JSON, including empty frames;
- H.264 High / yuv420p MP4 output;
- original audio preserved.

Default production parameters:

```text
detector         = ppocrv5_mobile
roi_bottom       = 0.45
ppocr_thresh     = 0.30
ppocr_box_thresh = 0.50
high_score       = 0.84
low_score        = 0.50
batch_size       = 32
decode_prefetch  = 4 batches
```

## Quick start — RTX/NVIDIA GPU

Create an environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

Install PyTorch CUDA 12.4, then the project/Paddle GPU runtime:

```bash
python -m pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
bash scripts/setup_ppocr_gpu.sh
```

PyTorch is still used by shared runtime/legacy FAST utilities. The measured
reference environment used PyTorch 2.6.0+cu124.

Run the production pipeline:

```bash
python scripts/run_subtitle_pipeline.py "AI Engineer test.mp4" \
  --output outputs/PP-OCRv5_mobile_PRODUCTION.mp4 \
  --export-coordinates outputs/PP-OCRv5_mobile_PRODUCTION.json \
  --detector ppocrv5_mobile \
  --device cuda \
  --batch-size 32 \
  --decode-prefetch-batches 4 \
  --roi-bottom-fraction .45 \
  --ppocr-thresh .30 \
  --ppocr-box-thresh .50 \
  --high-score .84 \
  --low-score .50
```

## CPU setup

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-ppocr-cpu.txt
```

Then use the same pipeline with `--device cpu`. For a new CPU machine,
benchmark batch size rather than assuming the RTX-3070 value is optimal.

## FAST fallback

FAST is retained as a fallback/reference backend:

```bash
bash scripts/setup_fast.sh

python scripts/run_subtitle_pipeline.py input.mp4 \
  --output outputs/FAST_fallback.mp4 \
  --detector fast \
  --device cuda \
  --temporal-mode v1
```

## Measured RTX 3070 production result

Measured on the supplied 3733-frame, 720x1280, 30 FPS video.

Environment:

```text
GPU:        NVIDIA GeForce RTX 3070
CPU:        Intel Xeon E5-2686 v4 @ 2.30 GHz
RAM:        ~62.8 GiB
OS:         Linux 6.8
Python:     3.10.12
Paddle:     3.3.1
PaddleOCR:  3.7.0
PyTorch:    2.6.0+cu124
CUDA:       12.4
Artifact source commit: f61be83cfe77fd9d755c402249321d60c15ee90d
```

Measured performance:

```text
Frames:                         3733
Detection loop:                 80.82 s
PP-OCR detector compute:        76.87 s
Detector throughput:            48.56 FPS
Temporal postprocess:            1.60 s
Render + H.264/audio encode:    37.00 s
End-to-end:                    119.49 s
End-to-end throughput:          31.24 FPS
Real-time factor:                0.960
Dropped frames:                  0
Multiline overlap frames:        0
Stable edge motion median/P95:   0 / 0 px
Coordinate records:           2581
```

Full-video regression against the existing FAST temporal reference:

```text
Reference records:             2581
Production records:            2581
Center precision @ 25 px:     1.000
Center recall @ 25 px:        1.000
Center F1 @ 25 px:            1.000
Missing boxed frames:             0
Ghost boxed frames:               0
```

Special regressions verified:
- frame 1105 remains empty (no transition ghost);
- frame 2552 is recovered by LOW-before-HIGH hysteresis;
- around 57 seconds, the short second-line glyph is fully enclosed and does
  not overlap line 1.

Measured final artifact hashes:

```text
MP4  de10801983cc62a221de08b89185db14ae202fb893ab95e7543932ead4c3a89d
JSON c2644d3b8d67b235fa6dc320d0c5ff42327d61d303668eb9b8944e4900629366
```

## Performance gate

The benchmark harness compares runtime variants on the same ROI/frames while
requiring coordinate quality to remain unchanged.

```bash
python scripts/benchmark_ppocr_performance_gate.py "AI Engineer test.mp4" \
  --golden-json artifacts/FAST_temporal_v1_consistent_final.json \
  --start-frame 1600 \
  --max-frames 300 \
  --batches 8,16,24,32 \
  --device gpu:0
```

Measured 300-frame gate around the difficult 57-second subtitle:

```text
static batch 8:       39.73 FPS
static batch 16:      41.79 FPS
static batch 24:      41.39 FPS
static batch 32:      41.95 FPS
threaded batch 16:    44.60 FPS
threaded batch 24:    45.66 FPS
threaded batch 32:    46.67 FPS  <- selected
dual predictor b16:   32.23 FPS
```

All successful variants above passed the coordinate quality gate. Two
concurrent GPU predictors were slower, so production uses one persistent
predictor plus a bounded decode-prefetch queue.

HPI, ONNX Runtime and TensorRT probes are represented in the benchmark tool.
In the measured environment they were not selected because their optional
runtime dependencies were not installed.

For ordinary detector/batch sweeps:

```bash
python scripts/benchmark_runtime.py "AI Engineer test.mp4" \
  --detector ppocrv5_mobile \
  --device cuda \
  --batches 16,24,32 \
  --precisions fp32 \
  --max-frames 600 \
  --output-dir outputs/bench_ppocr
```

## Tests

```bash
pytest -q
```

## Layout

- `src/video_text/ppocrv5_backend.py` — PP-OCRv5 Mobile detector backend
- `src/video_text/ppocr_temporal.py` — lightweight temporal/glyph geometry
- `src/video_text/pipeline.py` — production pipeline
- `scripts/run_subtitle_pipeline.py` — main CLI
- `scripts/benchmark_ppocr_performance_gate.py` — batch/thread/engine gate
- `scripts/benchmark_runtime.py` — repeatable runtime sweep
- `scripts/setup_ppocr_gpu.sh` — NVIDIA/Paddle GPU environment setup
- `requirements-ppocr-cpu.txt` — CPU Paddle runtime
- `tests/video_text/` — unit/integration/regression tests
- `artifacts/FAST_temporal_v1_consistent_final.json` — historical regression reference

## Third-party models

PP-OCRv5 is provided by PaddleOCR/PaddlePaddle. FAST is retained as a
fallback/reference backend and is provisioned at its pinned upstream commit
by `scripts/setup_fast.sh`.
