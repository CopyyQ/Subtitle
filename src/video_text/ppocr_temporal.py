from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .association import AssociationConfig, build_provisional_tracks, iou
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
