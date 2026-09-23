import numpy as np

from src.video_text.types import Candidate, SubtitleTrack, TrackObservation
from src.video_text.line_grouping import group_candidates_to_lines
from src.video_text.pipeline import PipelineConfig
from src.video_text.smoothing import (
    robust_track_envelope,
    expand_bbox_for_outline,
    synchronize_track_bbox,
)


def cand(box, score=.95, level="HIGH"):
    return Candidate(np.array(box, np.float32), score, level)


def make_track(boxes):
    t=SubtitleTrack(1, confirmed=True)
    for i,box in enumerate(boxes):
        t.observations[i]=TrackObservation(
            i, np.array(box,np.float32), .95, "HIGH"
        )
    return t


def test_thin_low_dash_attaches_to_confirmed_text_line():
    dash=cand([40,820,90,828], .62, "LOW")
    text=cand([100,800,400,840], .95, "HIGH")
    out=group_candidates_to_lines([dash,text])
    assert len(out)==1
    assert out[0].level=="HIGH"
    assert np.allclose(out[0].bbox,[40,800,400,840])


def test_same_baseline_components_merge_across_larger_missing_middle_gap():
    left=cand([100,800,180,840], .95, "HIGH")
    right=cand([300,801,400,841], .94, "HIGH")
    out=group_candidates_to_lines([left,right])
    assert len(out)==1
    assert np.allclose(out[0].bbox,[100,800,400,841])


def test_robust_envelope_keeps_width_seen_on_two_frames():
    t=make_track([
        [60,800,400,850],
        [60,800,400,850],
        [100,800,400,850],
        [100,800,400,850],
        [100,800,400,850],
        [100,800,400,850],
    ])
    box=robust_track_envelope(t)
    assert np.allclose(box,[60,800,400,850])


def test_robust_envelope_rejects_single_frame_spatial_outlier():
    t=make_track([
        [20,760,460,900],
        [100,800,400,850],
        [100,800,400,850],
        [100,800,400,850],
        [100,800,400,850],
        [100,800,400,850],
    ])
    box=robust_track_envelope(t)
    assert np.allclose(box,[100,800,400,850])


def test_outline_padding_scales_with_text_height_and_clamps():
    box=np.array([100,200,300,260],np.float32)
    out=expand_bbox_for_outline(
        box, frame_width=720, frame_height=1280,
        pad_ratio=.08, min_pad=3,
    )
    assert np.allclose(out,[95,195,305,265])

    edge=expand_bbox_for_outline(
        np.array([1,1,719,60],np.float32),
        frame_width=720, frame_height=1280,
        pad_ratio=.08, min_pad=3,
    )
    assert edge[0]==0 and edge[1]==0 and edge[2]==720


def test_synchronized_track_uses_padded_envelope_identically_every_frame():
    t=make_track([
        [60,800,400,850],
        [60,800,400,850],
        [100,800,400,850],
        [100,800,400,850],
        [100,800,400,850],
        [100,800,400,850],
    ])
    out=synchronize_track_bbox(
        t, frame_width=720, frame_height=1280,
        outline_pad_ratio=.08, min_outline_pad=3,
    )
    boxes=[out.observations[f].bbox for f in out.sorted_frames()]
    assert all(np.array_equal(boxes[0],b) for b in boxes[1:])
    assert np.allclose(boxes[0],[56,796,404,854])


def test_default_low_recall_configuration_keeps_thin_components():
    cfg=PipelineConfig()
    assert cfg.low_score <= .65
    assert cfg.low_min_area <= 40

def test_high_components_are_grouped_before_tiny_low_component():
    left=cand([188,852,233,896], .90, "HIGH")
    main=cand([231,849,530,903], .94, "HIGH")
    tiny_inside=cand([497,859,508,870], .66, "LOW")
    out=group_candidates_to_lines([tiny_inside,left,main])
    high=[x for x in out if x.level=="HIGH"]
    assert len(high)==1
    assert high[0].bbox[0] <= 188
    assert high[0].bbox[2] >= 530


def test_far_thin_low_noise_does_not_expand_confirmed_line():
    main=cand([256,831,463,895], .94, "HIGH")
    far_noise=cand([708,897,719,915], .62, "LOW")
    out=group_candidates_to_lines([main,far_noise])
    high=[x for x in out if x.level=="HIGH"]
    assert len(high)==1
    assert np.allclose(high[0].bbox,[256,831,463,895])
