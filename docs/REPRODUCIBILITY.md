# Reproducibility

This repository is the source snapshot associated with:
- FAST_temporal_v1_consistent_final.mp4
- FAST_temporal_v1_consistent_final.json

## Artifact fingerprints

JSON SHA-256:
2523abb714b14359b55d0d2cf62c4741ece33285710682dea9044ace7d991cf7

MP4 SHA-256:
aeb87a41b02426d5e4c9972c906c0a46a008fa76c1bd639bafa97ff6316a3318

The reference coordinate JSON is tracked under artifacts/. The MP4 is not
tracked in Git to keep repository history source-focused.

## Input contract

- 3733 frames
- 30 FPS
- 720 x 1280
- filename used during the run: AI Engineer test.mp4

The input video is intentionally not committed.

## FAST dependency

Repository: https://github.com/czczup/FAST.git

Pinned commit:
9cfeda29bfd16c4bc260740a5e39afb2cfdfd23f

Checkpoint:
fast_base_ic15_736_finetune_ic17mlt.pth

Checkpoint SHA-256:
e9f9f36dd343e04938c1758210f63c78494cf5b38efbf448138f18830bd42992

Run scripts/setup_fast.sh to reproduce the exact dependency location,
minimal import patch and checkpoint expected by the production code.

## Tested environment

- Python 3.10.20
- PyTorch 2.6.0+cu124
- TorchVision 0.21.0+cu124
- NumPy 2.2.6
- SciPy 1.15.3
- MMEngine 0.10.7
- EasyOCR 1.7.2
- pytest 8.4.2
- FFmpeg 4.3.1

CUDA 12.4 installation used for the reference run:

    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt

## Exact production command

    python scripts/run_subtitle_pipeline.py "AI Engineer test.mp4" --output outputs/FAST_temporal_v1_consistent_final.mp4 --temporal-mode v1

## Reference metrics

- frames: 3733
- boxed frames: 2216
- coordinate records: 2581
- logical tracks: 52
- multiline frames: 365
- merged fragmented tracks: 11
- merged groups: 8
- transition fragments absorbed: 1
- temporal-glyph tightened tracks: 28
- height-regularized tracks: 7
- OCR-tightened tracks: 3
- post-lock edge-range P95: 0.0 px
- post-lock max edge-range: 0.0 px
- multiline overlap frames: 0
- multiline overlap pixels: 0

Verify the tracked contract:

    python scripts/verify_consistent_final.py

If the reference MP4 is also available:

    python scripts/verify_consistent_final.py --video /path/to/FAST_temporal_v1_consistent_final.mp4

## Test modes

The repository does not commit the evaluation MP4. Therefore:

- without AI Engineer test.mp4 in the repository root, seven real-video
  integration tests are skipped;
- with that file present, the complete suite runs.

This keeps normal clones green without weakening the production-video
regressions.

## Repository layout note

The production repository uses a single detector only. The runtime model path
is `model/FAST` and generated media goes under `outputs/`.

## Hardware benchmark workflow

Use `scripts/benchmark_runtime.py` to measure real detector and end-to-end
throughput. Each variant performs an untimed FAST warm-up, deletes its own
detection cache, runs the requested frames, and records the exact command,
runtime configuration, provenance, and metrics. OOM/failure rows are retained
rather than silently reducing batch size.

Server RTX 3070 sweep:

```bash
python scripts/benchmark_runtime.py "AI Engineer test.mp4" \
  --device cuda --batches 4,8,16,32 --precisions fp32,fp16 \
  --max-frames 600 --output-dir outputs/bench_rtx3070
```

For CPU validation on Quyt, use the same tool with `--device cpu`,
`--precisions fp32`, and a CPU batch sweep. The exact selected measured
commands are copied into this document after the corresponding hardware sweep.
