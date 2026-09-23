import numpy as np

from src.video_text.types import Candidate, SubtitleTrack, TrackObservation
from src.video_text.line_grouping import suppress_nested_candidates
from src.video_text.lifecycle import split_tracks_on_geometry
from src.video_text.smoothing import synchronize_track_bbox


def c(b, score=.95, level="HIGH"):
    return Candidate(np.array(b, np.float32), score, level)


def track(tid, items):
    t=SubtitleTrack(tid, confirmed=True)
    for f,b in items.items():
        t.observations[f]=TrackObservation(f,np.array(b,np.float32),.95,"HIGH")
    return t


def test_nested_small_box_is_removed_but_separate_line_is_kept():
    large=c([80, 820, 640, 880], .96)
    nested=c([399, 833, 434, 847], .90)
    second_line=c([120, 900, 580, 950], .94)
    out=suppress_nested_candidates([large,nested,second_line])
    boxes=[tuple(x.bbox.tolist()) for x in out]
    assert tuple(large.bbox.tolist()) in boxes
    assert tuple(nested.bbox.tolist()) not in boxes
    assert tuple(second_line.bbox.tolist()) in boxes


def test_gap_same_geometry_stays_one_track():
    t=track(1,{10:[100,800,400,850],12:[102,800,402,850]})
    out=split_tracks_on_geometry([t],max_internal_gap=2)
    assert len(out)==1
    assert out[0].sorted_frames()==[10,12]


def test_gap_large_geometry_change_is_split_not_bridged():
    # Mirrors false bridge #1: width jumps from about 296 to 439 px.
    t=track(1,{10:[208,845,504,903],12:[140,841,579,910]})
    out=split_tracks_on_geometry([t],max_internal_gap=2)
    assert len(out)==2
    assert out[0].sorted_frames()==[10]
    assert out[1].sorted_frames()==[12]


def test_two_frame_gap_sentence_change_is_split():
    # Mirrors false bridge #7: width shrinks from about 500 to 380 px.
    t=track(1,{20:[109,845,609,903],23:[168,849,548,905]})
    out=split_tracks_on_geometry([t],max_internal_gap=2)
    assert len(out)==2


def test_synchronized_track_uses_one_identical_canonical_box():
    t=track(1,{
        1:[100,800,400,850],
        2:[102,801,401,851],
        3:[99,800,399,850],
        4:[101,799,400,849],
        5:[100,800,400,850],
    })
    out=synchronize_track_bbox(t)
    boxes=[out.observations[f].bbox for f in out.sorted_frames()]
    assert all(np.array_equal(boxes[0],b) for b in boxes[1:])
    assert np.array_equal(boxes[0],np.array([100,800,400,850],np.float32))
