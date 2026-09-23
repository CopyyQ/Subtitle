# Best CPU / GPU checkpoints

## CPU stable

- Branch: `feature/cpu-final-6412fe7`
- Commit: `6412fe7`
- Detector: PP-OCRv5 Mobile Det + OpenVINO CPU FP32
- Full 3733-frame best run: 42.19 FPS end-to-end
- Accuracy: 0 frame count mismatches; edge p95 1 px; edge max 2 px

Windows / Quyt:

```powershell
git fetch origin
git switch feature/cpu-final-6412fe7
.\scripts\run_cpu_final.ps1 -Video "D:\Temp\AI Engineer test.mp4" -Model "outputs\openvino_quyt\model.xml" -OutputDir "outputs\cpu_final"
```

## RTX 3070 GPU stable

- Branch: `feature/rtx3070-gpu`
- Acceleration commit: `cb852d8`
- Detector: same PP-OCRv5 Mobile Det model as CPU
- Runtime: ONNX Runtime CUDA with fused uint8 NHWC preprocessing
- Gate / ROI / thresholds are the same as CPU final

Full 3733-frame RTX 3070 validation:

- 85.77 FPS end-to-end
- 181.50 detector FPS
- detection loop: 23.10 s
- detector: 20.57 s
- render/encode: 19.05 s
- 2581 / 2581 final boxes
- 0 frame count mismatches vs CPU stable
- edge p95: 0 px
- edge max: 1 px
- dropped frames: 0

Linux / RTX 3070:

```bash
git fetch origin
git switch feature/rtx3070-gpu
PYTHON_BIN=/home/plab/Desktop/Nguyen_Anh_Quyet_PLAB/AI/Test/Subtitle_gpu/.venv_gpu/bin/python \
  bash scripts/run_gpu_best.sh \
  "/home/plab/Desktop/Nguyen_Anh_Quyet_PLAB/AI/Test/AI Engineer test.mp4" \
  outputs/gpu_best
```

Required optimized GPU runtime packages on the validated server: `onnxruntime-gpu==1.23.2`, `onnx==1.17.0`, `paddlepaddle-gpu==3.3.1`, `paddleocr==3.7.0`, `paddlex==3.7.2`.
