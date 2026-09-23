# Subtitle V1 - FAST Temporal Chinese Hard-Subtitle Detector

Production-oriented detector for hard-coded Chinese subtitles in video.
It uses FAST for scene-text proposals and a temporal, line-aware geometry
pipeline to keep bounding boxes tight and stable over each subtitle episode.

This repository is the source snapshot associated with
FAST_temporal_v1_consistent_final.mp4.

## Main behavior

- line-level boxes, including independent boxes for wrapped second lines;
- logical subtitle stitching across fragmented detector tracks;
- cross-fade and transition-fragment handling;
- canonical static geometry per subtitle line;
- temporal glyph/outline tightening with OCR safety guards;
- multiline non-overlap and center consistency;
- weak and single-character subtitle recovery;
- MP4 plus coordinate JSON output;
- original audio preserved through ffmpeg remux.

Reference run: 52 logical tracks, 2581 coordinate records, 0 px post-lock
edge-range P95, and 0 multiline overlap frames.

## Quick start

1. Create an environment:

    python -m venv .venv
    source .venv/bin/activate

2. Install PyTorch CUDA 12.4 as used in the reference run:

    pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
    pip install -r requirements.txt

3. Install the pinned FAST dependency and checkpoint:

    bash scripts/setup_fast.sh

4. Run:

    python scripts/run_subtitle_pipeline.py input.mp4 --output outputs/subtitle_bbox.mp4 --temporal-mode v1

Reference command:

    python scripts/run_subtitle_pipeline.py "AI Engineer test.mp4" --output outputs/FAST_temporal_v1_consistent_final.mp4 --temporal-mode v1

5. Test:

    pytest -q

6. Verify the tracked reference coordinates:

    python scripts/verify_consistent_final.py

## Layout

- src/video_text/ - production pipeline
- scripts/run_subtitle_pipeline.py - CLI
- scripts/setup_fast.sh - pinned FAST/checkpoint setup
- scripts/verify_consistent_final.py - artifact contract verifier
- patches/ - exact FAST compatibility patch
- tests/video_text/ - unit/integration/regression tests
- artifacts/ - reference coordinate JSON
- docs/ARCHITECTURE.md - design
- docs/REPRODUCIBILITY.md - versions, hashes and exact run command
- docs/assignment_vi.txt - original task statement

## Reference artifact hashes

JSON:
2523abb714b14359b55d0d2cf62c4741ece33285710682dea9044ace7d991cf7

MP4:
aeb87a41b02426d5e4c9972c906c0a46a008fa76c1bd639bafa97ff6316a3318

See docs/REPRODUCIBILITY.md for the full provenance record.

## Third-party model

FAST is developed at https://github.com/czczup/FAST and is provisioned at a
pinned upstream commit by scripts/setup_fast.sh. The upstream Apache-2.0
license remains authoritative for FAST code.
