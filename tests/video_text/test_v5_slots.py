import cv2
import numpy as np

from src.video_text.v5_slots import bootstrap_line_slots


def _draw_outlined_rect(frame,x1,y1,x2,y2):
    cv2.rectangle(frame,(x1-3,y1-3),(x2+3,y2+3),(10,10,10),-1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(245,245,245),-1)


def _two_line_frames(n=7):
    frames=[]
    for _ in range(n):
        f=np.full((240,360,3),100,np.uint8)
        # wide first row
        for x in range(65,286,38):
            _draw_outlined_rect(f,x,80,x+22,112)
        # one-character second row, centered
        _draw_outlined_rect(f,165,126,195,158)
        frames.append(f)
    return frames


def _one_line_frames(n=7):
    frames=[]
    for _ in range(n):
        f=np.full((240,360,3),100,np.uint8)
        for x in range(85,276,38):
            _draw_outlined_rect(f,x,104,x+22,138)
        frames.append(f)
    return frames


def test_bootstrap_locks_two_vertical_slots_from_first_frames():
    slots=bootstrap_line_slots(
        _two_line_frames(),
        search_bbox=[50,65,310,180],
    )
    assert len(slots)==2
    assert slots[0].y1 <= 77
    assert slots[0].y2 >= 115
    assert slots[1].y1 <= 123
    assert slots[1].y2 >= 161
    assert slots[0].y2 < slots[1].y2


def test_bootstrap_keeps_short_second_line_as_real_slot():
    slots=bootstrap_line_slots(
        _two_line_frames(),
        search_bbox=[50,65,310,180],
    )
    assert len(slots)==2
    assert abs(slots[1].expected_x_center-180) <= 1


def test_bootstrap_returns_one_slot_for_single_line():
    slots=bootstrap_line_slots(
        _one_line_frames(),
        search_bbox=[60,80,300,160],
    )
    assert len(slots)==1
    assert slots[0].y1 <= 101
    assert slots[0].y2 >= 141


def test_transient_second_row_in_one_frame_does_not_create_slot():
    frames=_one_line_frames()
    _draw_outlined_rect(frames[2],165,150,195,180)
    slots=bootstrap_line_slots(
        frames,
        search_bbox=[60,80,300,195],
    )
    assert len(slots)==1

def test_persistent_short_background_run_below_does_not_create_second_slot():
    frames=[]
    for _ in range(7):
        f=np.full((240,360,3),100,np.uint8)
        for x in range(85,276,38):
            _draw_outlined_rect(f,x,92,x+22,134)
        # Persistent outlined background artifact, but only half font height.
        _draw_outlined_rect(f,165,151,195,167)
        frames.append(f)
    slots=bootstrap_line_slots(
        frames,
        search_bbox=[60,75,300,190],
    )
    assert len(slots)==1
    assert slots[0].y2 < 150
