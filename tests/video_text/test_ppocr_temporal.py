import numpy as np

from src.video_text.ppocr_temporal import (
    PPOCRTemporalConfig,
    build_ppocr_temporal_tracks,
)
from src.video_text.types import Candidate, FrameDetections


def cand(box, score=.95, level="HIGH"):
    return Candidate(np.asarray(box, np.float32), score, level, "ppocrv5_mobile")


def fd(frame, high=(), low=()):
    return FrameDetections(
        frame_index=frame,
        timestamp=frame / 30.0,
        high=list(high),
        low=list(low),
    )


def frames_of(track):
    return track.sorted_frames()


def test_low_before_high_is_backfilled_into_confirmed_track():
    frames = [
        fd(0, low=[cand([200, 880, 520, 930], .65, "LOW")]),
        fd(1, high=[cand([202, 881, 522, 931])]),
        fd(2, high=[cand([201, 880, 521, 930])]),
    ]

    tracks, metrics = build_ppocr_temporal_tracks(
        frames, 720, 1280, PPOCRTemporalConfig()
    )

    assert len(tracks) == 1
    assert frames_of(tracks[0]) == [0, 1, 2]
    assert tracks[0].observations[0].level == "LOW"
    assert metrics["ppocr_low_backfill_count"] == 1


def test_low_only_noise_does_not_create_track():
    frames = [
        fd(0, low=[cand([20, 600, 80, 630], .70, "LOW")]),
        fd(1, low=[cand([21, 600, 81, 630], .71, "LOW")]),
        fd(2, low=[cand([20, 601, 80, 631], .72, "LOW")]),
    ]

    tracks, _ = build_ppocr_temporal_tracks(
        frames, 720, 1280, PPOCRTemporalConfig()
    )

    assert tracks == []


def test_internal_one_frame_gap_is_filled_for_same_geometry():
    frames = [
        fd(0, high=[cand([190, 880, 530, 930])]),
        fd(1, high=[cand([191, 880, 531, 930])]),
        fd(2),
        fd(3, high=[cand([190, 881, 530, 931])]),
        fd(4, high=[cand([190, 880, 530, 930])]),
    ]

    tracks, metrics = build_ppocr_temporal_tracks(
        frames, 720, 1280, PPOCRTemporalConfig()
    )

    assert len(tracks) == 1
    assert frames_of(tracks[0]) == [0, 1, 2, 3, 4]
    assert tracks[0].observations[2].reconstructed is True
    assert tracks[0].observations[2].level == "PP_GAP"
    assert metrics["ppocr_gap_fill_count"] == 1


def test_geometry_change_across_blank_frame_preserves_transition_gap():
    frames = [
        fd(0, high=[cand([220, 880, 500, 930])]),
        fd(1, high=[cand([220, 880, 500, 930])]),
        fd(2, high=[cand([220, 880, 500, 930])]),
        fd(3),
        fd(4, high=[cand([80, 880, 640, 930])]),
        fd(5, high=[cand([80, 880, 640, 930])]),
        fd(6, high=[cand([80, 880, 640, 930])]),
    ]

    tracks, metrics = build_ppocr_temporal_tracks(
        frames, 720, 1280, PPOCRTemporalConfig()
    )

    assert len(tracks) == 2
    assert all(3 not in frames_of(track) for track in tracks)
    assert metrics["ppocr_transition_gap_preserved_count"] >= 1


def test_stable_high_text_outside_dominant_subtitle_band_is_filtered():
    frames = []
    for i in range(12):
        frames.append(
            fd(
                i,
                high=[
                    cand([180, 875, 540, 925]),
                    cand([20, 560, 170, 600], .96),
                ],
            )
        )

    tracks, metrics = build_ppocr_temporal_tracks(
        frames, 720, 1280, PPOCRTemporalConfig()
    )

    assert len(tracks) == 1
    cy = np.median([
        (obs.bbox[1] + obs.bbox[3]) / 2
        for obs in tracks[0].observations.values()
    ])
    assert cy > 800
    assert metrics["ppocr_out_of_band_rejected_count"] == 1
