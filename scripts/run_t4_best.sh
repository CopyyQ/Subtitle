#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
VIDEO="${1:-/content/AI Engineer test.mp4}"
OUTDIR="${2:-outputs/t4_best}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${PPOCR_ONNX_MODEL:-$ROOT/models/PP-OCRv5_mobile_det/inference.onnx}"
TRT_CACHE="${PPOCR_TRT_CACHE_DIR:-$HOME/.cache/video_text/ppocrv5_trt_t4}"
[[ -f "$MODEL" ]] || { echo "Missing model: $MODEL" >&2; exit 2; }
TRT_LIBS="$($PYTHON_BIN - <<'PY'
import glob,os,sys
paths=[]
for root in sys.path:
    paths.extend(glob.glob(os.path.join(root,'tensorrt_libs')))
    paths.extend(glob.glob(os.path.join(root,'nvidia','*','lib')))
print(':'.join(dict.fromkeys(paths)))
PY
)"
export LD_LIBRARY_PATH="${TRT_LIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
$PYTHON_BIN - <<'PY'
import onnxruntime as ort, tensorrt as trt
providers=ort.get_available_providers()
required={'TensorrtExecutionProvider','CUDAExecutionProvider'}
missing=required-set(providers)
if missing: raise SystemExit(f'Missing GPU providers: {sorted(missing)}; available={providers}')
print('TensorRT',trt.__version__,'providers',providers)
PY
ffmpeg -hide_banner -encoders 2>/dev/null | grep h264_nvenc >/dev/null || { echo 'h264_nvenc unavailable' >&2; exit 3; }
mkdir -p "$OUTDIR" "$TRT_CACHE"
$PYTHON_BIN scripts/prebuild_gpu_tensorrt.py "$VIDEO" --model "$MODEL" --cache-dir "$TRT_CACHE" --batch-size 64 --precision fp32
$PYTHON_BIN scripts/run_subtitle_pipeline.py "$VIDEO" \
  --output "$OUTDIR/out.mp4" --output-encoder nvenc_direct --encode-preset ultrafast \
  --detector ppocrv5_mobile --device cuda --batch-size 64 --decode-prefetch-batches 4 \
  --roi-bottom-fraction .45 --ppocr-gpu-engine tensorrt --ppocr-trt-precision fp32 \
  --ppocr-trt-cache-dir "$TRT_CACHE" --ppocr-adaptive-gating \
  --ppocr-gate-max-skip-frames 12 --ppocr-gate-change-threshold .015 \
  --ppocr-gate-bright-net-threshold .0075 --ppocr-thresh .30 --ppocr-box-thresh .50 \
  --high-score .84 --low-score .50 --export-coordinates "$OUTDIR/out.json"
