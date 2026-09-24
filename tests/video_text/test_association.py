import numpy as np
from src.video_text.types import Candidate, FrameDetections
from src.video_text.association import AssociationConfig, build_provisional_tracks

def cand(b, score=.95, level="HIGH"):
    return Candidate(np.array(b,np.float32),score,level)

def fd(i, high=None, low=None):
    return FrameDetections(i,i/30.0,high or [],low or [])

def cfg():
    return AssociationConfig(iou_gate=.20,center_gate=.04,
        min_width_ratio=.65,max_width_ratio=1.55,
        min_height_ratio=.65,max_height_ratio=1.55,
        max_frame_gap=3,backfill_frames=2)

def test_high_can_start_track():
    tracks=build_provisional_tracks([fd(0,high=[cand([100,800,400,850])])],cfg())
    assert len(tracks)==1
    assert tracks[0].confirmed

def test_low_cannot_start_track():
    assert build_provisional_tracks([fd(0,low=[cand([100,800,400,850],.78,"LOW")])],cfg())==[]

def test_low_continues_existing_high_track():
    frames=[
        fd(0,high=[cand([100,800,400,850])]),
        fd(1,low=[cand([102,800,402,850],.78,"LOW")]),
    ]
    tracks=build_provisional_tracks(frames,cfg())
    assert tracks[0].sorted_frames()==[0,1]
    assert tracks[0].observations[1].level=="LOW"

def test_far_high_after_gap_starts_new_track():
    frames=[
        fd(0,high=[cand([100,800,400,850])]),
        fd(1),
        fd(2,high=[cand([10,650,150,700])]),
    ]
    tracks=build_provisional_tracks(frames,cfg())
    assert len(tracks)==2

def test_low_before_high_is_backfilled_but_does_not_exist_alone():
    frames=[
        fd(0,low=[cand([100,800,400,850],.78,"LOW")]),
        fd(1,high=[cand([101,800,401,850],.94,"HIGH")]),
        fd(2,high=[cand([102,800,402,850],.95,"HIGH")]),
    ]
    tracks=build_provisional_tracks(frames,cfg())
    assert len(tracks)==1
    assert tracks[0].sorted_frames()==[0,1,2]
    assert tracks[0].observations[0].level=="LOW"


def test_multiple_pending_low_candidates_on_same_frame_do_not_compare_numpy_boxes():
    frames=[
        fd(0,low=[
            cand([100,800,400,850],.78,"LOW"),
            cand([500,900,620,940],.76,"LOW"),
        ]),
        fd(1,high=[cand([101,800,401,850],.94,"HIGH")]),
    ]
    tracks=build_provisional_tracks(frames,cfg())
    assert len(tracks)==1
    assert tracks[0].sorted_frames()==[0,1]
