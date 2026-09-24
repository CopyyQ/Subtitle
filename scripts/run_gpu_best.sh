#!/usr/bin/env bash
set -euo pipefail

VIDEO="${1:-/home/plab/Desktop/Nguyen_Anh_Quyet_PLAB/AI/Test/AI Engineer test.mp4}"
OUTDIR="${2:-outputs/gpu_best}"
PYTHON_BIN="${PYTHON_BIN:-python}"
MODEL="${PPOCR_ONNX_MODEL:-$HOME/.paddlex/official_models/PP-OCRv5_mobile_det/inference.onnx}"
TRT_CACHE="${PPOCR_TRT_CACHE_DIR:-$HOME/.cache/video_text/ppocrv5_trt}"
TENSORRT_LIB_DIR="${TENSORRT_LIB_DIR:-/usr/local/TensorRT/lib}"

if [[ ! -f "$MODEL" ]]; then
  echo "Missing PP-OCRv5 ONNX model: $MODEL" >&2
  exit 2
fi

NVLIBS="$($PYTHON_BIN - <<'PY'
import glob,os,sys
paths=[]
for root in sys.path:
    paths.extend(glob.glob(os.path.join(root,'nvidia','*','lib')))
print(':'.join(dict.fromkeys(paths)))
PY
)"
export LD_LIBRARY_PATH="$TENSORRT_LIB_DIR${NVLIBS:+:$NVLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

"$PYTHON_BIN" - <<'PY'
import onnxruntime as ort
providers=ort.get_available_providers()
required={"TensorrtExecutionProvider","CUDAExecutionProvider"}
missing=required-set(providers)
if missing:
    raise SystemExit(f"GPU providers unavailable: missing={sorted(missing)} available={providers}")
print("onnxruntime providers:",providers)
PY

mkdir -p "$OUTDIR" "$TRT_CACHE"
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

"$PYTHON_BIN" scripts/prebuild_gpu_tensorrt.py "$VIDEO" \
  --model "$MODEL" --cache-dir "$TRT_CACHE" \
  --batch-size 64 --precision fp32

"$PYTHON_BIN" scripts/run_subtitle_pipeline.py "$VIDEO" \
  --output "$OUTDIR/out.mp4" \
  --encode-preset ultrafast \
  --detector ppocrv5_mobile \
  --device cuda \
  --batch-size 64 \
  --decode-prefetch-batches 4 \
  --roi-bottom-fraction .45 \
  --ppocr-gpu-engine tensorrt \
  --ppocr-trt-precision fp32 \
  --ppocr-trt-cache-dir "$TRT_CACHE" \
  --ppocr-adaptive-gating \
  --ppocr-gate-max-skip-frames 1 \
  --ppocr-gate-change-threshold .015 \
  --ppocr-gate-bright-net-threshold .0075 \
  --ppocr-thresh .30 \
  --ppocr-box-thresh .50 \
  --high-score .84 \
  --low-score .50 \
  --export-coordinates "$OUTDIR/out.json"
