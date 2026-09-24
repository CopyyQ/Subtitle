from pathlib import Path

import cv2
import numpy as np

from src.video_text.ppocr_temporal import (
    PPOCRTemporalConfig,
    clamp_ppocr_multiline_seams,
    glyph_safe_short_bbox,
    recover_ppocr_short_lines,
)
from src.video_text.types import (
    Candidate,
    FrameDetections,
    SubtitleTrack,
    TrackObservation,
)


WORK = Path(__file__).resolve().parent / ".tmp_ppocr_glyph"


def make_track(track_id, box, frames=range(8), level="HIGH"):
    track = SubtitleTrack(track_id, confirmed=True)
    for f in frames:
        track.observations[f] = TrackObservation(
            f, np.asarray(box, np.float32), .95, level
        )
    return track


def test_short_line_bbox_encloses_visible_glyph_support_without_overlapping_parent():
    observed = [
        [349, 891, 368, 920],
        [354, 897, 361, 910],
        [354, 894, 361, 917],
    ]
    glyph_extents = [
        [340, 882, 379, 924],
        [340, 882, 379, 924],
    ]
    parent = [79, 821, 639, 880]

    box = glyph_safe_short_bbox(
        observed_boxes=observed,
        glyph_extents=glyph_extents,
        parent_bbox=parent,
        frame_shape=(1280, 720),
    )

    assert box[0] <= 340
    assert box[1] <= 882
    assert box[2] >= 379
    assert box[3] >= 924
    assert box[1] >= parent[3] + 1


def test_single_extreme_glyph_extent_does_not_inflate_short_box():
    observed = [
        [349, 891, 368, 920],
        [350, 890, 369, 921],
        [349, 891, 368, 920],
    ]
    glyph_extents = [
        [340, 882, 379, 924],
        [341, 882, 380, 924],
        [340, 883, 379, 925],
        [100, 700, 650, 1000],
    ]

    box = glyph_safe_short_bbox(
        observed_boxes=observed,
        glyph_extents=glyph_extents,
        parent_bbox=[79, 821, 639, 880],
        frame_shape=(1280, 720),
    )

    assert box[0] > 300
    assert box[2] < 430
    assert box[1] > 850
    assert box[3] < 960


def test_short_low_line_persists_below_confirmed_parent_and_uses_glyph_extent():
    WORK.mkdir(parents=True, exist_ok=True)
    source = WORK / "short_line.mp4"
    writer = cv2.VideoWriter(
        str(source),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (720, 1280),
    )
    assert writer.isOpened()
    frames = []
    for f in range(8):
        image = np.zeros((1280, 720, 3), np.uint8)
        cv2.rectangle(image, (340, 882), (378, 923), (255, 255, 255), -1)
        writer.write(image)
        frames.append(
            FrameDetections(
                f,
                f / 10.0,
                high=[],
                low=[
                    Candidate(
                        np.array([350, 891, 368, 919], np.float32),
                        .60,
                        "LOW",
                        "ppocrv5_mobile",
                    )
                ],
            )
        )
    writer.release()

    parent = make_track(1, [80, 821, 640, 880])
    recovered, metrics = recover_ppocr_short_lines(
        source,
        [parent],
        frames,
        frame_width=720,
        frame_height=1280,
        config=PPOCRTemporalConfig(),
    )

    assert len(recovered) == 1
    assert recovered[0].sorted_frames() == list(range(8))
    box = recovered[0].observations[0].bbox
    assert box[0] <= 340
    assert box[1] <= 882
    assert box[2] >= 379
    assert box[3] >= 924
    assert box[1] >= 881
    assert metrics["ppocr_weak_short_line_count"] == 1


def test_multiline_seam_removes_overlap_and_keeps_track_geometry_constant():
    top = make_track(1, [80, 821, 640, 888], frames=range(4))
    bottom = make_track(2, [336, 881, 384, 927], frames=range(4))

    metrics = clamp_ppocr_multiline_seams([top, bottom], min_gap=1)

    top_boxes = [top.observations[f].bbox for f in top.sorted_frames()]
    bottom_boxes = [bottom.observations[f].bbox for f in bottom.sorted_frames()]
    assert all(np.array_equal(top_boxes[0], box) for box in top_boxes[1:])
    assert all(np.array_equal(bottom_boxes[0], box) for box in bottom_boxes[1:])
    assert top_boxes[0][3] < bottom_boxes[0][1]
    assert bottom_boxes[0][1] <= 882
    assert metrics["ppocr_multiline_overlap_frame_count"] == 0
