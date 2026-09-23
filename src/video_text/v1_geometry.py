from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .text_enhancement import white_black_text_mask


@dataclass(slots=True)
class TemporalGeometryLock:
    boxes: list[np.ndarray]
    canonical_bbox: np.ndarray
    moving: bool
    prelock_max_edge_range_px: float


def _as_boxes(boxes):
    arr = np.asarray(boxes, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 4 or len(arr) == 0:
        raise ValueError("boxes must be a non-empty Nx4 sequence")
    if np.any(arr[:, 2] <= arr[:, 0]) or np.any(arr[:, 3] <= arr[:, 1]):
        raise ValueError("boxes must have positive area")
    return arr


def detect_consistent_translation(
    boxes,
    *,
    min_center_range_px=4.0,
    max_size_range_px=2.5,
):
    arr = _as_boxes(boxes)
    widths = arr[:, 2] - arr[:, 0]
    heights = arr[:, 3] - arr[:, 1]
    cxs = (arr[:, 0] + arr[:, 2]) * 0.5
    cys = (arr[:, 1] + arr[:, 3]) * 0.5

    moving_x = (
        float(np.ptp(cxs)) >= float(min_center_range_px)
        and float(np.ptp(widths)) <= float(max_size_range_px)
    )
    moving_y = (
        float(np.ptp(cys)) >= float(min_center_range_px)
        and float(np.ptp(heights)) <= float(max_size_range_px)
    )
    return bool(moving_x or moving_y)


def _clip_box(box, width, height):
    out = np.asarray(box, dtype=np.float32).copy()
    out[0] = np.clip(out[0], 0.0, float(width))
    out[2] = np.clip(out[2], 0.0, float(width))
    out[1] = np.clip(out[1], 0.0, float(height))
    out[3] = np.clip(out[3], 0.0, float(height))
    return out


def _fallback_median_bbox(arr):
    med = np.median(arr, axis=0).astype(np.float32)
    if med[2] <= med[0] or med[3] <= med[1]:
        raise ValueError("median bbox has non-positive area")
    return med


def clamp_to_temporal_anchor(candidate, boxes, *, max_expand_px=1.0):
    arr = _as_boxes(boxes)
    anchor = _fallback_median_bbox(arr)
    out = np.asarray(candidate, dtype=np.float32).copy()
    expand = max(0.0, float(max_expand_px))

    # Expansion is allowed only on an edge for which the temporal sequence
    # itself contains support outside the robust median anchor. A perfectly
    # stable V5.5 edge must never grow merely because scene pixels inside the
    # search crop look text-like.
    left_expand = min(
        expand,
        max(0.0, float(anchor[0]) - float(np.min(arr[:, 0]))),
    )
    top_expand = min(
        expand,
        max(0.0, float(anchor[1]) - float(np.min(arr[:, 1]))),
    )
    right_expand = min(
        expand,
        max(0.0, float(np.max(arr[:, 2])) - float(anchor[2])),
    )
    bottom_expand = min(
        expand,
        max(0.0, float(np.max(arr[:, 3])) - float(anchor[3])),
    )
    out[0] = max(float(out[0]), float(anchor[0]) - left_expand)
    out[1] = max(float(out[1]), float(anchor[1]) - top_expand)
    out[2] = min(float(out[2]), float(anchor[2]) + right_expand)
    out[3] = min(float(out[3]), float(anchor[3]) + bottom_expand)
    if out[2] <= out[0] or out[3] <= out[1]:
        return anchor.copy()
    return out


def temporal_canonical_bbox(
    frames,
    boxes,
    *,
    vote_threshold=.45,
    pad_px=2,
):
    arr = _as_boxes(boxes)
    frames = list(frames)
    if len(frames) != len(arr):
        raise ValueError("frames and boxes must have identical length")
    if not frames:
        raise ValueError("frames must not be empty")

    h, w = frames[0].shape[:2]
    for frame in frames:
        if frame.shape[:2] != (h, w):
            raise ValueError("all frames must share dimensions")

    # Build a robust search window from box quantiles, not min/max, so a
    # single noisy frame cannot enlarge the search region.
    qlo = np.percentile(arr[:, [0, 1]], 10, axis=0)
    qhi = np.percentile(arr[:, [2, 3]], 90, axis=0)
    guard = max(3, int(round(float(pad_px) + 2)))
    cx1 = max(0, int(np.floor(qlo[0])) - guard)
    cy1 = max(0, int(np.floor(qlo[1])) - guard)
    cx2 = min(w, int(np.ceil(qhi[0])) + guard)
    cy2 = min(h, int(np.ceil(qhi[1])) + guard)
    if cx2 <= cx1 or cy2 <= cy1:
        return _fallback_median_bbox(arr)

    votes = np.zeros((cy2 - cy1, cx2 - cx1), dtype=np.float32)
    for frame in frames:
        mask = white_black_text_mask(frame[cy1:cy2, cx1:cx2])
        votes += (mask > 0).astype(np.float32)
    persistent = (votes / max(1, len(frames))) >= float(vote_threshold)

    if not np.any(persistent):
        return _fallback_median_bbox(arr)

    binary = (persistent.astype(np.uint8) * 255)
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)
    median_box = _fallback_median_bbox(arr)
    median_h = float(median_box[3] - median_box[1])
    target_cy = .5 * float(median_box[1] + median_box[3])
    min_area = max(3.0, .0015 * median_h * median_h)
    min_h = max(2.0, .12 * median_h)

    selected = []
    for i in range(1, n):
        x, y, wc, hc, area = stats[i]
        if float(area) < min_area:
            continue
        abs_cy = cy1 + float(centroids[i][1])
        center_error = abs(abs_cy - target_cy)

        if float(hc) < min_h:
            # Some real CJK glyphs/punctuation (for example 一) collapse to
            # a persistent horizontal stroke only a few pixels high. Do not
            # discard them merely because component height is small. Require
            # stronger width/area/persistence geometry instead so random
            # speckles and short background edges still fail the gate.
            min_short_width = .30 * median_h
            min_short_area = max(min_area, .015 * median_h * median_h)
            if (
                float(wc) < min_short_width
                or float(area) < min_short_area
                or center_error > .25 * median_h
            ):
                continue
        elif center_error > .55 * median_h:
            continue

        selected.append(
            (
                float(cx1 + x),
                float(cy1 + y),
                float(cx1 + x + wc),
                float(cy1 + y + hc),
            )
        )

    if not selected:
        return _fallback_median_bbox(arr)

    box = np.array(
        [
            min(b[0] for b in selected) - float(pad_px),
            min(b[1] for b in selected) - float(pad_px),
            max(b[2] for b in selected) + float(pad_px),
            max(b[3] for b in selected) + float(pad_px),
        ],
        dtype=np.float32,
    )
    box = _clip_box(box, w, h)

    # Persistent evidence is trusted for tightening, but do not let it jump
    # implausibly outside the robust temporal envelope.
    envelope = np.array(
        [
            float(np.percentile(arr[:, 0], 10)) - guard,
            float(np.percentile(arr[:, 1], 10)) - guard,
            float(np.percentile(arr[:, 2], 90)) + guard,
            float(np.percentile(arr[:, 3], 90)) + guard,
        ],
        dtype=np.float32,
    )
    envelope = _clip_box(envelope, w, h)
    box[0] = max(box[0], envelope[0])
    box[1] = max(box[1], envelope[1])
    box[2] = min(box[2], envelope[2])
    box[3] = min(box[3], envelope[3])

    box = clamp_to_temporal_anchor(
        box,
        arr,
        max_expand_px=1.0,
    )
    if box[2] <= box[0] or box[3] <= box[1]:
        return _fallback_median_bbox(arr)
    return box


def _median_filter(values, window=3):
    values = np.asarray(values, dtype=np.float32)
    radius = max(0, int(window) // 2)
    out = []
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        out.append(float(np.median(values[lo:hi])))
    return np.asarray(out, dtype=np.float32)


def lock_line_geometry(
    frames,
    boxes,
    *,
    vote_threshold=.45,
    pad_px=2,
):
    arr = _as_boxes(boxes)
    moving = detect_consistent_translation(arr)
    prelock_max_edge_range = float(np.max(np.ptp(arr, axis=0)))

    if not moving:
        canonical = temporal_canonical_bbox(
            frames,
            arr,
            vote_threshold=vote_threshold,
            pad_px=pad_px,
        )
        return TemporalGeometryLock(
            boxes=[canonical.copy() for _ in range(len(arr))],
            canonical_bbox=canonical.copy(),
            moving=False,
            prelock_max_edge_range_px=prelock_max_edge_range,
        )

    widths = arr[:, 2] - arr[:, 0]
    heights = arr[:, 3] - arr[:, 1]
    width = float(np.median(widths))
    height = float(np.median(heights))
    cxs = _median_filter((arr[:, 0] + arr[:, 2]) * .5, window=3)
    cys = _median_filter((arr[:, 1] + arr[:, 3]) * .5, window=3)

    locked = [
        np.array(
            [
                float(cx - width * .5),
                float(cy - height * .5),
                float(cx + width * .5),
                float(cy + height * .5),
            ],
            dtype=np.float32,
        )
        for cx, cy in zip(cxs, cys)
    ]
    canonical = np.median(np.stack(locked, axis=0), axis=0).astype(np.float32)
    return TemporalGeometryLock(
        boxes=locked,
        canonical_bbox=canonical,
        moving=True,
        prelock_max_edge_range_px=prelock_max_edge_range,
    )
