import cv2
import numpy as np

from src.video_text.v5_weak_birth import SlotPrior
from src.video_text.v5_extent import consensus_slot_bbox


def _outlined_rect(frame,x1,y1,x2,y2):
    cv2.rectangle(frame,(x1-3,y1-3),(x2+3,y2+3),(10,10,10),-1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(245,245,245),-1)


def _frames(with_noise=False,with_thin_leader=False):
    out=[]
    for i in range(7):
        f=np.full((220,360,3),100,np.uint8)
        if with_thin_leader:
            _outlined_rect(f,55,104,88,109)
        for x in (105,145,185,225):
            _outlined_rect(f,x,92,x+25,126)
        if with_noise and i==2:
            _outlined_rect(f,10,90,42,128)
        out.append(f)
    return out


def test_temporal_consensus_ignores_one_frame_horizontal_noise():
    slot=SlotPrior(y1=84,y2=134,expected_x_center=180,expected_height=42)
    box=consensus_slot_bbox(_frames(with_noise=True),slot)

    assert box is not None
    assert box[0] > 80
    assert box[2] < 270
    assert box[1] == 84
    assert box[3] == 134


def test_stable_thin_leading_glyph_is_included():
    slot=SlotPrior(y1=84,y2=134,expected_x_center=180,expected_height=42)
    box=consensus_slot_bbox(_frames(with_thin_leader=True),slot)

    assert box is not None
    # Stable thin leader begins at x=55; outline-safe bbox must include it.
    assert box[0] <= 52
    assert box[2] >= 250


def test_single_centered_glyph_produces_compact_consensus_box():
    frames=[]
    for _ in range(7):
        f=np.full((220,360,3),80,np.uint8)
        _outlined_rect(f,164,95,196,130)
        frames.append(f)
    slot=SlotPrior(y1=86,y2=138,expected_x_center=180,expected_height=44)

    box=consensus_slot_bbox(frames,slot)

    assert box is not None
    assert 150 <= box[0] <= 162
    assert 198 <= box[2] <= 210
    assert box[1] == 86
    assert box[3] == 138


def test_empty_slot_returns_none():
    frames=[np.full((220,360,3),100,np.uint8) for _ in range(7)]
    slot=SlotPrior(y1=84,y2=134,expected_x_center=180,expected_height=42)
    assert consensus_slot_bbox(frames,slot) is None
