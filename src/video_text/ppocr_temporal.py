from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .association import AssociationConfig, build_provisional_tracks, iou
from .text_enhancement import white_black_text_mask
from .types import SubtitleTrack, TrackObservation


@dataclass(slots=True)
class PPOCRTemporalConfig:
    max_internal_gap: int = 2
    stable_frames: int = 8
    reference_height: int = 1280
    subtitle_band_half_height_px: float = 70.0
    same_geometry_iou: float = .55
    same_geometry_width_ratio: float = 1.25
    same_geometry_height_ratio: float = 1.25
    same_geometry_center_px: float = 22.0


def _center(box):
    box = np.asarray(box, np.float32)
    return (box[:2] + box[2:]) * .5


def _ratio_symmetric(a: float, b: float) -> float:
    a = max(float(a), 1.0)
    b = max(float(b), 1.0)
    return max(a, b) / min(a, b)


def same_subtitle_geometry(
    a,
    b,
    *,
    frame_height: int = 1280,
    config: PPOCRTemporalConfig | None = None,
) -> bool:
    cfg = config or PPOCRTemporalConfig()
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    wa, wb = a[2] - a[0], b[2] - b[0]
    ha, hb = a[3] - a[1], b[3] - b[1]
    if _ratio_symmetric(wa, wb) > cfg.same_geometry_width_ratio:
        return False
    if _ratio_symmetric(ha, hb) > cfg.same_geometry_height_ratio:
        return False
    scale = max(float(frame_height), 1.0) / float(cfg.reference_height)
    if float(np.linalg.norm(_center(a) - _center(b))) > cfg.same_geometry_center_px * scale:
        return False
    return iou(a, b) >= cfg.same_geometry_iou


def _copy_observation(obs: TrackObservation) -> TrackObservation:
    return TrackObservation(
        obs.frame_index,
        obs.bbox.copy(),
        obs.score,
        obs.level,
        obs.reconstructed,
    )


def _split_geometry_segments(
    tracks: list[SubtitleTrack],
    *,
    frame_height: int,
    config: PPOCRTemporalConfig,
) -> tuple[list[SubtitleTrack], int]:
    segments: list[SubtitleTrack] = []
    transition_splits = 0
    next_id = 1
    for track in tracks:
        frames = track.sorted_frames()
        current: list[TrackObservation] = []
        previous: TrackObservation | None = None
        for fi in frames:
            obs = track.observations[fi]
            split = False
            if previous is not None:
                distance = fi - previous.frame_index
                if distance > config.max_internal_gap + 1:
                    split = True
                elif not same_subtitle_geometry(
                    previous.bbox,
                    obs.bbox,
                    frame_height=frame_height,
                    config=config,
                ):
                    split = True
                    if 1 < distance <= config.max_internal_gap + 1:
                        transition_splits += 1
            if split and current:
                seg = SubtitleTrack(next_id, confirmed=True)
                next_id += 1
                seg.observations = {
                    item.frame_index: _copy_observation(item)
                    for item in current
                }
                segments.append(seg)
                current = []
            current.append(obs)
            previous = obs
        if current:
            seg = SubtitleTrack(next_id, confirmed=True)
            next_id += 1
            seg.observations = {
                item.frame_index: _copy_observation(item)
                for item in current
            }
            segments.append(seg)
    return segments, transition_splits


def _fill_internal_gaps(
    tracks: list[SubtitleTrack],
    *,
    frame_height: int,
    config: PPOCRTemporalConfig,
) -> int:
    filled = 0
    for track in tracks:
        observed_frames = track.sorted_frames()
        for left_f, right_f in zip(observed_frames, observed_frames[1:]):
            missing = right_f - left_f - 1
            if not 1 <= missing <= config.max_internal_gap:
                continue
            left = track.observations[left_f]
            right = track.observations[right_f]
            if not same_subtitle_geometry(
                left.bbox,
                right.bbox,
                frame_height=frame_height,
                config=config,
            ):
                continue
            for step in range(1, missing + 1):
                alpha = step / float(missing + 1)
                bbox = (
                    (1.0 - alpha) * left.bbox
                    + alpha * right.bbox
                ).astype(np.float32)
                fi = left_f + step
                track.observations[fi] = TrackObservation(
                    fi,
                    bbox,
                    None,
                    "PP_GAP",
                    reconstructed=True,
                )
                filled += 1
    return filled


def _track_center_y(track: SubtitleTrack) -> float:
    return float(np.median([
        _center(obs.bbox)[1]
        for obs in track.observations.values()
        if not obs.reconstructed
    ]))


def _dominant_band(
    tracks: list[SubtitleTrack],
    *,
    frame_height: int,
    config: PPOCRTemporalConfig,
) -> tuple[float, float] | None:
    stable = []
    for track in tracks:
        observed = [
            obs for obs in track.observations.values()
            if not obs.reconstructed
        ]
        if len(observed) < config.stable_frames:
            continue
        stable.append((_track_center_y(track), len(observed)))
    if not stable:
        return None

    scale = max(float(frame_height), 1.0) / float(config.reference_height)
    half = config.subtitle_band_half_height_px * scale
    scored = []
    for center, _ in stable:
        cluster = [
            (cy, weight)
            for cy, weight in stable
            if abs(cy - center) <= half
        ]
        support = sum(weight for _, weight in cluster)
        weighted_center = sum(cy * weight for cy, weight in cluster) / max(support, 1)
        scored.append((support, weighted_center, cluster))
    # Ties choose the lower-on-screen cluster, which is the usual subtitle band.
    _, center, _ = max(scored, key=lambda x: (x[0], x[1]))
    return center - half, center + half


def _count_preserved_transition_gaps(
    tracks: list[SubtitleTrack],
    *,
    frame_height: int,
    config: PPOCRTemporalConfig,
) -> int:
    spans = []
    for track in tracks:
        fs = track.sorted_frames()
        if not fs:
            continue
        spans.append((min(fs), max(fs), track))
    spans.sort(key=lambda x: x[0])
    count = 0
    for (_, end, left), (start, _, right) in zip(spans, spans[1:]):
        missing = start - end - 1
        if not 1 <= missing <= config.max_internal_gap:
            continue
        a = left.observations[max(left.observations)].bbox
        b = right.observations[min(right.observations)].bbox
        if not same_subtitle_geometry(
            a, b, frame_height=frame_height, config=config
        ):
            count += 1
    return count


def build_ppocr_temporal_tracks(
    frames,
    frame_width: int,
    frame_height: int,
    config: PPOCRTemporalConfig | None = None,
):
    cfg = config or PPOCRTemporalConfig()
    association = AssociationConfig(
        iou_gate=.20,
        center_gate=.04,
        min_width_ratio=.65,
        max_width_ratio=1.55,
        min_height_ratio=.65,
        max_height_ratio=1.55,
        # AssociationConfig measures observed-frame distance. A value of
        # three is required to bridge at most two missing frames.
        max_frame_gap=cfg.max_internal_gap + 1,
        backfill_frames=2,
    )
    provisional = build_provisional_tracks(frames, association)
    low_count = sum(
        1
        for track in provisional
        for obs in track.observations.values()
        if obs.level == "LOW"
    )
    tracks, transition_splits = _split_geometry_segments(
        provisional,
        frame_height=frame_height,
        config=cfg,
    )
    transition_preserved = transition_splits + _count_preserved_transition_gaps(
        tracks,
        frame_height=frame_height,
        config=cfg,
    )
    gap_count = _fill_internal_gaps(
        tracks,
        frame_height=frame_height,
        config=cfg,
    )

    band = _dominant_band(
        tracks,
        frame_height=frame_height,
        config=cfg,
    )
    rejected = 0
    if band is not None:
        lo, hi = band
        kept = []
        for track in tracks:
            cy = _track_center_y(track)
            if lo <= cy <= hi:
                kept.append(track)
            else:
                rejected += 1
        tracks = kept

    # Renumber only after all filters so output IDs stay compact.
    for tid, track in enumerate(tracks, 1):
        track.track_id = tid

    metrics = {
        "ppocr_track_count": len(tracks),
        "ppocr_low_backfill_count": low_count,
        "ppocr_gap_fill_count": gap_count,
        "ppocr_transition_gap_preserved_count": transition_preserved,
        "ppocr_out_of_band_rejected_count": rejected,
        "ppocr_dominant_band_y1": None if band is None else float(band[0]),
        "ppocr_dominant_band_y2": None if band is None else float(band[1]),
    }
    return tracks, metrics


def _robust_box_rows(boxes):
    arr = np.asarray(boxes, np.float32).reshape(-1, 4)
    if len(arr) <= 2:
        return arr
    centers = (arr[:, :2] + arr[:, 2:]) * .5
    widths = np.maximum(1.0, arr[:, 2] - arr[:, 0])
    heights = np.maximum(1.0, arr[:, 3] - arr[:, 1])
    med_center = np.median(centers, axis=0)
    med_width = max(float(np.median(widths)), 1.0)
    med_height = max(float(np.median(heights)), 1.0)
    center_limit = max(18.0, 1.75 * med_height)
    dist = np.linalg.norm(centers - med_center, axis=1)
    width_ratio = np.maximum(widths / med_width, med_width / widths)
    height_ratio = np.maximum(heights / med_height, med_height / heights)
    keep = (
        (dist <= center_limit)
        & (width_ratio <= 3.0)
        & (height_ratio <= 3.0)
    )
    filtered = arr[keep]
    return filtered if len(filtered) else arr


def _box_union(boxes):
    arr = _robust_box_rows(boxes)
    return np.array(
        [
            float(arr[:, 0].min()),
            float(arr[:, 1].min()),
            float(arr[:, 2].max()),
            float(arr[:, 3].max()),
        ],
        dtype=np.float32,
    )


def measure_short_line_glyph_extent(frame, search_box):
    box = np.asarray(search_box, np.float32)
    h, w = frame.shape[:2]
    x1 = max(0, int(np.floor(box[0])))
    y1 = max(0, int(np.floor(box[1])))
    x2 = min(w, int(np.ceil(box[2])))
    y2 = min(h, int(np.ceil(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    mask = white_black_text_mask(crop)
    ys, xs = np.where(mask > 0)
    if not len(xs):
        return None
    return np.array(
        [
            float(xs.min() + x1),
            float(ys.min() + y1),
            float(xs.max() + 1 + x1),
            float(ys.max() + 1 + y1),
        ],
        dtype=np.float32,
    )


def glyph_safe_short_bbox(
    observed_boxes,
    glyph_extents,
    parent_bbox,
    frame_shape,
):
    if not observed_boxes:
        raise ValueError("observed_boxes must not be empty")
    observed = _box_union(observed_boxes)
    glyph = _box_union(glyph_extents) if glyph_extents else None
    support = observed.copy()
    if glyph is not None:
        support[0] = min(support[0], glyph[0])
        support[1] = min(support[1], glyph[1])
        support[2] = max(support[2], glyph[2])
        support[3] = max(support[3], glyph[3])

    glyph_height = max(
        1.0,
        float((glyph if glyph is not None else support)[3] - (glyph if glyph is not None else support)[1]),
    )
    pad_x = max(4, int(round(glyph_height * .10)))
    pad_y = max(3, int(round(glyph_height * .07)))
    box = support + np.array([-pad_x, -pad_y, pad_x, pad_y], np.float32)

    frame_height, frame_width = int(frame_shape[0]), int(frame_shape[1])
    box[0] = np.clip(box[0], 0, frame_width)
    box[2] = np.clip(box[2], 0, frame_width)
    box[1] = np.clip(box[1], 0, frame_height)
    box[3] = np.clip(box[3], 0, frame_height)

    if parent_bbox is not None:
        parent = np.asarray(parent_bbox, np.float32)
        seam_floor = float(parent[3]) + 1.0
        glyph_top = float(glyph[1]) if glyph is not None else float(support[1])
        if glyph_top >= seam_floor:
            box[1] = max(float(box[1]), seam_floor)
    return box.astype(np.float32)


def _canonical_bbox(track: SubtitleTrack) -> np.ndarray:
    return np.median(
        np.stack([obs.bbox for obs in track.observations.values()]),
        axis=0,
    ).astype(np.float32)


def _track_span(track: SubtitleTrack) -> tuple[int, int]:
    fs = track.sorted_frames()
    return min(fs), max(fs)


def _weak_low_tracks(
    frames,
    *,
    frame_width: int,
    frame_height: int,
    config: PPOCRTemporalConfig,
):
    xscale = max(float(frame_width), 1.0) / 720.0
    yscale = max(float(frame_height), 1.0) / 1280.0
    selected = {}
    for frame in frames:
        rows = []
        for cand in frame.low:
            box = np.asarray(cand.bbox, np.float32)
            center = _center(box)
            width = float(box[2] - box[0])
            height = float(box[3] - box[1])
            if abs(float(center[0]) - frame_width * .5) > 70.0 * xscale:
                continue
            if width > 140.0 * xscale or height > 90.0 * yscale:
                continue
            rows.append(cand)
        if rows:
            selected[frame.frame_index] = rows

    tracks = []
    active = []
    max_gap_distance = 4
    center_gate = 24.0 * max(xscale, yscale)
    for fi in sorted(selected):
        active = [
            idx for idx in active
            if fi - tracks[idx]["last"] <= max_gap_distance
        ]
        candidates = selected[fi]
        pairs = []
        for idx in active:
            previous = tracks[idx]["obs"][-1][1].bbox
            pc = _center(previous)
            for j, cand in enumerate(candidates):
                distance = float(np.linalg.norm(pc - _center(cand.bbox)))
                if distance <= center_gate:
                    pairs.append((distance, idx, j))
        used_tracks = set()
        used_candidates = set()
        for _, idx, j in sorted(pairs):
            if idx in used_tracks or j in used_candidates:
                continue
            tracks[idx]["obs"].append((fi, candidates[j]))
            tracks[idx]["last"] = fi
            used_tracks.add(idx)
            used_candidates.add(j)
        for j, cand in enumerate(candidates):
            if j in used_candidates:
                continue
            tracks.append({"obs": [(fi, cand)], "last": fi})
            active.append(len(tracks) - 1)
    return tracks


def _representative_indices(indices, limit=12):
    indices = sorted(set(int(x) for x in indices))
    if len(indices) <= limit:
        return indices
    pick = np.linspace(0, len(indices) - 1, limit, dtype=int)
    return [indices[int(i)] for i in pick]


def _glyph_extents_for_weak_track(source, weak_obs, search_box):
    indices = _representative_indices([fi for fi, _ in weak_obs])
    if not indices:
        return []
    cap = cv2.VideoCapture(str(Path(source)))
    extents = []
    try:
        for fi in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, frame = cap.read()
            if not ok:
                continue
            extent = measure_short_line_glyph_extent(frame, search_box)
            if extent is not None:
                extents.append(extent)
    finally:
        cap.release()
    return extents


def recover_ppocr_short_lines(
    source,
    strong_tracks,
    frames,
    frame_width: int,
    frame_height: int,
    config: PPOCRTemporalConfig | None = None,
):
    cfg = config or PPOCRTemporalConfig()
    xscale = max(float(frame_width), 1.0) / 720.0
    yscale = max(float(frame_height), 1.0) / 1280.0

    parents = []
    for track in strong_tracks:
        fs = track.sorted_frames()
        if not fs:
            continue
        box = _canonical_bbox(track)
        if float(box[2] - box[0]) < 180.0 * xscale:
            continue
        start, end = min(fs), max(fs)
        parents.append((track, start, end, box))

    weak_sequences = _weak_low_tracks(
        frames,
        frame_width=frame_width,
        frame_height=frame_height,
        config=cfg,
    )
    recovered = []
    next_id = max([t.track_id for t in strong_tracks] + [0]) + 1
    for weak in weak_sequences:
        obs = weak["obs"]
        if len(obs) < 3:
            continue
        start, end = obs[0][0], obs[-1][0]
        if end - start < 4:
            continue
        observed_boxes = [cand.bbox for _, cand in obs]
        weak_center = np.median(
            np.stack([_center(box) for box in observed_boxes]),
            axis=0,
        )
        matches = []
        for parent, pstart, pend, pbox in parents:
            overlap = max(0, min(end, pend) - max(start, pstart) + 1)
            if overlap <= 0:
                continue
            pc = _center(pbox)
            if abs(float(weak_center[0] - pc[0])) > 80.0 * xscale:
                continue
            dy = float(weak_center[1] - pc[1])
            if not 15.0 * yscale <= dy <= 90.0 * yscale:
                continue
            if start - pstart > 12:
                continue
            matches.append((overlap, parent, pstart, pend, pbox))
        if not matches:
            continue

        _, parent, pstart, pend, pbox = max(matches, key=lambda x: x[0])
        observed_union = _box_union(observed_boxes)
        median_height = float(np.median([
            max(1.0, float(box[3] - box[1]))
            for box in observed_boxes
        ]))
        expand_x = max(12.0 * xscale, .75 * median_height)
        expand_y = max(8.0 * yscale, .35 * median_height)
        search = observed_union + np.array(
            [-expand_x, -expand_y, expand_x, expand_y],
            np.float32,
        )
        glyph_extents = _glyph_extents_for_weak_track(
            source,
            obs,
            search,
        )
        canonical = glyph_safe_short_bbox(
            observed_boxes,
            glyph_extents,
            pbox,
            (frame_height, frame_width),
        )
        score = float(np.median([cand.score for _, cand in obs]))
        track = SubtitleTrack(next_id, confirmed=True)
        next_id += 1
        for fi in range(pstart, pend + 1):
            track.observations[fi] = TrackObservation(
                fi,
                canonical.copy(),
                score,
                "PP_WEAK_HOLD",
                reconstructed=True,
            )
        recovered.append(track)

    return recovered, {
        "ppocr_weak_short_line_count": len(recovered),
        "ppocr_weak_short_line_frame_count": sum(
            len(track.observations) for track in recovered
        ),
    }


def clamp_ppocr_multiline_seams(
    tracks: list[SubtitleTrack],
    min_gap: int = 1,
):
    adjustments = 0
    for i, first in enumerate(tracks):
        if not first.observations:
            continue
        first_start, first_end = _track_span(first)
        first_box = _canonical_bbox(first)
        for second in tracks[i + 1:]:
            if not second.observations:
                continue
            second_start, second_end = _track_span(second)
            if min(first_end, second_end) < max(first_start, second_start):
                continue
            second_box = _canonical_bbox(second)
            if _center(first_box)[1] <= _center(second_box)[1]:
                top, bottom = first, second
                top_box, bottom_box = first_box, second_box
            else:
                top, bottom = second, first
                top_box, bottom_box = second_box, first_box

            if float(top_box[3]) < float(bottom_box[1]):
                continue
            if abs(float(_center(top_box)[0] - _center(bottom_box)[0])) > 120.0:
                continue

            top_h = max(1.0, float(top_box[3] - top_box[1]))
            bottom_h = max(1.0, float(bottom_box[3] - bottom_box[1]))
            top_w = max(1.0, float(top_box[2] - top_box[0]))
            bottom_w = max(1.0, float(bottom_box[2] - bottom_box[0]))

            if bottom_h < .75 * top_h or bottom_w < .50 * top_w:
                top_y2 = min(float(top_box[3]), float(bottom_box[1]) - min_gap)
                bottom_y1 = float(bottom_box[1])
            else:
                seam = np.floor(
                    (float(top_box[3]) + float(bottom_box[1]) - min_gap) * .5
                )
                top_y2 = float(seam)
                bottom_y1 = float(seam + min_gap)

            if top_y2 <= float(top_box[1]) + 4:
                continue
            if bottom_y1 >= float(bottom_box[3]) - 4:
                continue
            for obs in top.observations.values():
                obs.bbox[3] = top_y2
            for obs in bottom.observations.values():
                obs.bbox[1] = bottom_y1
            adjustments += 1
            first_box = _canonical_bbox(first)

    overlap_frames = set()
    by_frame = {}
    for track in tracks:
        for fi, obs in track.observations.items():
            by_frame.setdefault(fi, []).append(obs.bbox)
    for fi, boxes in by_frame.items():
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i], boxes[j]
                inter_w = max(0.0, min(float(a[2]), float(b[2])) - max(float(a[0]), float(b[0])))
                inter_h = max(0.0, min(float(a[3]), float(b[3])) - max(float(a[1]), float(b[1])))
                if inter_w * inter_h > 0:
                    overlap_frames.add(fi)

    return {
        "ppocr_multiline_seam_adjustment_count": adjustments,
        "ppocr_multiline_overlap_frame_count": len(overlap_frames),
    }
