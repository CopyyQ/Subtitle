import cv2
import numpy as np

from src.video_text.v1_geometry import (
    clamp_to_temporal_anchor,
    detect_consistent_translation,
    lock_line_geometry,
    temporal_canonical_bbox,
)


def _frame_with_text_box(
    *,
    size=(240, 360),
    glyph_box=(80, 120, 280, 166),
):
    h, w = size
    frame = np.full((h, w, 3), 110, np.uint8)
    x1, y1, x2, y2 = glyph_box
    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (8, 8, 8), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 245, 245), -1)
    # Break the rectangle into glyph-like vertical groups so the temporal
    # mask cannot pass merely because one giant solid component exists.
    for x in range(x1 + 18, x2, 28):
        cv2.rectangle(frame, (x, y1 - 2), (x + 7, y2 + 2), (110, 110, 110), -1)
    return frame


def test_static_noisy_boxes_lock_to_one_identical_canonical_bbox():
    frames = [_frame_with_text_box() for _ in range(9)]
    boxes = [
        [76, 116, 284, 170],
        [78, 118, 282, 169],
        [74, 117, 284, 168],
        [79, 118, 282, 168],
        [75, 116, 283, 170],
        [80, 118, 282, 168],
        [77, 117, 284, 169],
        [75, 118, 283, 168],
        [78, 116, 282, 170],
    ]
    result = lock_line_geometry(frames, boxes, pad_px=2)
    assert result.moving is False
    rounded = [tuple(int(round(v)) for v in b) for b in result.boxes]
    assert len(set(rounded)) == 1
    assert result.prelock_max_edge_range_px >= 4.0


def test_temporal_canonical_bbox_is_tighter_than_noisy_union_without_clipping_text():
    frames = [_frame_with_text_box() for _ in range(11)]
    boxes = [
        [70 + (i % 4), 112 + (i % 3), 290 - (i % 2), 174 - (i % 3)]
        for i in range(11)
    ]
    canonical = temporal_canonical_bbox(frames, boxes, pad_px=2)
    x1, y1, x2, y2 = [int(round(v)) for v in canonical]
    # Visible outline is approximately [78,118,282,168].
    assert x1 <= 80
    assert y1 <= 120
    assert x2 >= 280
    assert y2 >= 166

    union = [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]
    canonical_area = (x2 - x1) * (y2 - y1)
    union_area = (union[2] - union[0]) * (union[3] - union[1])
    assert canonical_area < union_area


def test_one_frame_scene_noise_does_not_expand_static_canonical_box():
    frames = [_frame_with_text_box() for _ in range(9)]
    noisy = frames[4].copy()
    cv2.rectangle(noisy, (35, 105), (55, 178), (250, 250, 250), -1)
    frames[4] = noisy
    boxes = [[70, 110, 290, 175] for _ in frames]
    canonical = temporal_canonical_bbox(frames, boxes, pad_px=2)
    assert canonical[0] > 55


def test_true_constant_size_translation_is_not_frozen():
    boxes = [[80 + i, 120, 280 + i, 166] for i in range(8)]
    assert detect_consistent_translation(boxes) is True

    frames = [
        _frame_with_text_box(glyph_box=(80 + i, 120, 280 + i, 166))
        for i in range(8)
    ]
    result = lock_line_geometry(frames, boxes, pad_px=2)
    assert result.moving is True
    centers = [
        ((b[0] + b[2]) * .5, (b[1] + b[3]) * .5)
        for b in result.boxes
    ]
    assert centers[-1][0] > centers[0][0] + 4
    widths = [round(float(b[2] - b[0]), 4) for b in result.boxes]
    heights = [round(float(b[3] - b[1]), 4) for b in result.boxes]
    assert len(set(widths)) == 1
    assert len(set(heights)) == 1


def test_one_edge_jitter_is_not_misclassified_as_real_translation():
    boxes = [
        [80, 120, 280, 166],
        [82, 120, 280, 166],
        [84, 120, 280, 166],
        [86, 120, 280, 166],
        [84, 120, 280, 166],
        [82, 120, 280, 166],
        [80, 120, 280, 166],
    ]
    assert detect_consistent_translation(boxes) is False


def test_persistent_thin_horizontal_leading_glyph_is_not_dropped():
    frames=[]
    for _ in range(9):
        frame=np.full((240,360,3),110,np.uint8)
        # A glyph such as Chinese 一 / a thin horizontal component can be
        # only a few pixels high but is persistent and semantically real.
        cv2.rectangle(frame,(92,139),(136,144),(245,245,245),-1)
        cv2.rectangle(frame,(90,137),(138,146),(8,8,8),2)
        _outlined_glyph_band = None
        cv2.rectangle(frame,(150,120),(282,166),(8,8,8),-1)
        cv2.rectangle(frame,(152,122),(280,164),(245,245,245),-1)
        frames.append(frame)
    boxes=[[86,116,286,170] for _ in frames]
    canonical=temporal_canonical_bbox(frames,boxes,pad_px=2)
    assert canonical[0] <= 92, canonical


def test_persistent_evidence_cannot_expand_beyond_trusted_temporal_anchor():
    frames=[
        _frame_with_text_box(glyph_box=(96,116,264,174))
        for _ in range(9)
    ]
    boxes=[[100,120,260,170] for _ in frames]
    canonical=temporal_canonical_bbox(frames,boxes,pad_px=2)
    # V1 may tighten the robust V5.5 temporal anchor, but it may expand
    # outside that anchor by at most one safety pixel. Persistent scene/text
    # evidence outside the trusted lifecycle envelope must not bloat the box.
    assert canonical[0] >= 99, canonical
    assert canonical[1] >= 119, canonical
    assert canonical[2] <= 261, canonical
    assert canonical[3] <= 171, canonical


def test_candidate_bbox_expands_only_on_temporally_supported_edges():
    boxes=[
        [100,120,260,170],
        [101,119,260,170],
        [100,120,259,171],
        [100,120,260,170],
        [100,120,260,170],
    ]
    candidate=np.array([92,112,268,178],np.float32)
    clamped=clamp_to_temporal_anchor(candidate,boxes,max_expand_px=1)
    assert np.allclose(clamped,[100,119,260,171])

def test_stable_temporal_anchor_does_not_expand_without_edge_support():
    boxes=[[100,120,260,170] for _ in range(7)]
    candidate=np.array([96,116,264,174],np.float32)
    clamped=clamp_to_temporal_anchor(candidate,boxes,max_expand_px=1)
    assert np.allclose(clamped,[100,120,260,170])
