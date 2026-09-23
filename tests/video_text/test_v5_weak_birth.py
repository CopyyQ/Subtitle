import cv2
import numpy as np

from src.video_text.v5_weak_birth import (
    SlotPrior,
    WeakBirthTracker,
    detect_weak_text_in_slot,
)


def _frame_with_centered_glyph():
    frame=np.full((200,320,3),60,np.uint8)
    # dark outline
    cv2.rectangle(frame,(145,82),(177,126),(10,10,10),-1)
    # white core split into two parts to mimic a Chinese glyph
    cv2.rectangle(frame,(150,86),(172,108),(245,245,245),-1)
    cv2.rectangle(frame,(156,108),(170,121),(245,245,245),-1)
    return frame


def test_slot_pixel_detector_recovers_single_centered_glyph_without_fast_box():
    prior=SlotPrior(y1=75,y2=135,expected_x_center=160,expected_height=45)
    box=detect_weak_text_in_slot(_frame_with_centered_glyph(),prior)
    assert box is not None
    x1,y1,x2,y2=box
    assert x1 <= 145
    assert y1 <= 82
    assert x2 >= 177
    assert y2 >= 126


def test_white_region_without_dark_outline_is_rejected():
    frame=np.full((200,320,3),120,np.uint8)
    cv2.rectangle(frame,(145,82),(177,126),(245,245,245),-1)
    prior=SlotPrior(y1=75,y2=135,expected_x_center=160,expected_height=45)
    assert detect_weak_text_in_slot(frame,prior) is None


def test_persistent_weak_birth_confirms_after_three_frames_and_backfills():
    tracker=WeakBirthTracker(confirm_frames=3,max_gap=0)
    box=np.array([145,82,177,126],np.float32)

    assert tracker.update(10,box) == []
    assert tracker.update(11,box) == []
    rows=tracker.update(12,box)

    assert [r.frame for r in rows] == [10,11,12]
    assert all(r.confirmed for r in rows)


def test_one_frame_weak_noise_never_becomes_subtitle():
    tracker=WeakBirthTracker(confirm_frames=3,max_gap=0)
    box=np.array([145,82,177,126],np.float32)
    assert tracker.update(10,box) == []
    assert tracker.update(11,None) == []
    assert tracker.flush() == []


def test_blank_gap_starts_new_weak_segment_instead_of_bridging_old_one():
    tracker=WeakBirthTracker(confirm_frames=3,max_gap=0)
    a=np.array([140,82,180,126],np.float32)
    b=np.array([145,82,177,126],np.float32)

    tracker.update(1,a)
    tracker.update(2,a)
    first=tracker.update(3,a)
    assert [r.frame for r in first] == [1,2,3]

    assert tracker.update(4,None) == []
    assert tracker.update(5,None) == []
    assert tracker.update(6,b) == []
    assert tracker.update(7,b) == []
    second=tracker.update(8,b)

    assert [r.frame for r in second] == [6,7,8]
    assert all(r.frame > 5 for r in second)

def test_confirmed_weak_track_does_not_absorb_spatially_different_text():
    tracker=WeakBirthTracker(confirm_frames=3,max_gap=0)
    a=np.array([145,82,177,126],np.float32)
    b=np.array([220,82,252,126],np.float32)

    tracker.update(1,a)
    tracker.update(2,a)
    first=tracker.update(3,a)
    assert [r.frame for r in first]==[1,2,3]

    # A new subtitle at a different x position must not inherit A's canonical
    # bbox merely because it appears in the next frame.
    assert tracker.update(4,b)==[]
    assert tracker.update(5,b)==[]
    second=tracker.update(6,b)
    assert [r.frame for r in second]==[4,5,6]
    assert all(r.bbox[0] >= 215 for r in second)
