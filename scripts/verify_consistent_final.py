from __future__ import annotations
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

EXPECTED_JSON_SHA256 = "2523abb714b14359b55d0d2cf62c4741ece33285710682dea9044ace7d991cf7"
EXPECTED_VIDEO_SHA256 = "aeb87a41b02426d5e4c9972c906c0a46a008fa76c1bd639bafa97ff6316a3318"

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def verify_json(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    meta = data["metadata"]
    records = data["records"]

    assert meta["temporal_mode"] == "v1"
    assert meta["frames"] == 3733
    assert meta["boxed_frames"] == 2216
    assert meta["track_count"] == 52
    assert len(records) == 2581
    assert meta["v1_merged_group_count"] == 8
    assert meta["v1_merged_track_count"] == 11
    assert meta["v1_transition_fragment_absorbed_count"] == 1
    assert meta["v1_glyph_tightened_track_count"] == 28
    assert meta["v1_height_regularized_track_count"] == 7
    assert meta["v1_ocr_tightened_track_count"] == 3
    assert meta["v1_postlock_edge_range_p95_px"] == 0.0
    assert meta["v1_postlock_max_edge_range_px"] == 0.0
    assert meta["multiline_overlap_frame_count"] == 0
    assert meta["multiline_overlap_pixel_count"] == 0

    by_track = defaultdict(list)
    by_frame = defaultdict(list)
    for row in records:
        by_track[int(row["track_id"])].append(row)
        by_frame[int(row["frame"])].append(row)

    for tid, rows in by_track.items():
        boxes = {tuple(row["bbox"]) for row in rows}
        assert len(boxes) == 1, (tid, boxes)

    assert sum(len(rows) > 1 for rows in by_frame.values()) == 365

    for frame in (71, 72, 73, 170, 171, 172, 246, 247, 248, 1105, 3238, 3239):
        assert not by_frame[frame], (frame, by_frame[frame])

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="artifacts/FAST_temporal_v1_consistent_final.json")
    parser.add_argument("--video")
    args = parser.parse_args()

    json_path = Path(args.json)
    json_sha = sha256_file(json_path)
    if json_path.name == "FAST_temporal_v1_consistent_final.json":
        assert json_sha == EXPECTED_JSON_SHA256, json_sha

    verify_json(json_path)
    print(f"JSON OK: {json_path}")
    print(f"JSON SHA256: {json_sha}")

    if args.video:
        video_sha = sha256_file(Path(args.video))
        assert video_sha == EXPECTED_VIDEO_SHA256, video_sha
        print(f"VIDEO SHA256 OK: {video_sha}")

    print("CONSISTENT_FINAL_VERIFIED")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
