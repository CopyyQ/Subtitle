#!/usr/bin/env bash
set -euo pipefail

VIDEO="${1:-/home/plab/Desktop/Nguyen_Anh_Quyet_PLAB/AI/Test/AI Engineer test.mp4}"
OUTDIR="${2:-outputs/gpu_best}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${PPOCR_ONNX_MODEL:-$HOME/.paddlex/official_models/PP-OCRv5_mobile_det/inference.onnx}"

if [[ ! -f "$MODEL" ]]; then
  echo "Missing PP-OCRv5 ONNX model: $MODEL" >&2
  echo "Prepare the PaddleOCR/HPI model cache before running the optimized GPU path." >&2
  exit 2
fi

"$PYTHON_BIN" - <<'PY'
import onnxruntime as ort
providers = ort.get_available_providers()
if "CUDAExecutionProvider" not in providers:
    raise SystemExit(f"CUDAExecutionProvider unavailable: {providers}")
print("onnxruntime providers:", providers)
PY

mkdir -p "$OUTDIR"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON_BIN" scripts/run_subtitle_pipeline.py "$VIDEO" \
  --output "$OUTDIR/out.mp4" \
  --encode-preset ultrafast \
  --detector ppocrv5_mobile \
  --device cuda \
  --batch-size 64 \
  --decode-prefetch-batches 4 \
  --roi-bottom-fraction .45 \
  --ppocr-adaptive-gating \
  --ppocr-gate-max-skip-frames 1 \
  --ppocr-gate-change-threshold .015 \
  --ppocr-gate-bright-net-threshold .0075 \
  --ppocr-thresh .30 \
  --ppocr-box-thresh .50 \
  --high-score .84 \
  --low-score .50 \
  --export-coordinates "$OUTDIR/out.json"
