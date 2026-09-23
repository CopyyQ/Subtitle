import cv2
import numpy as np

from src.video_text.v5_extent import consensus_slot_bbox
from src.video_text.v5_weak_birth import SlotPrior


def _outlined_rect(frame,x1,y1,x2,y2):
    cv2.rectangle(frame,(x1-3,y1-3),(x2+3,y2+3),(15,15,15),-1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(225,225,225),-1)


def test_coarse_supported_slot_blocks_large_non_text_edge_expansion():
    frames=[]
    for _ in range(7):
        f=np.full((180,420,3),105,np.uint8)
        # Real text starts at x=90.
        for x in (90,130,170,210,250,290):
            _outlined_rect(f,x,72,x+25,112)
        # Bright/dark background attached to the first glyph and spanning the
        # full slot height. Enhancement must not pull the bbox to x=40.
        cv2.rectangle(f,(40,66),(87,118),(35,35,35),-1)
        cv2.rectangle(f,(45,66),(84,118),(180,180,180),-1)
        frames.append(f)

    slot=SlotPrior(
        y1=66,y2=118,expected_x_center=205,expected_height=52,
        expected_x1=85,expected_x2=320,
    )
    box=consensus_slot_bbox(frames,slot)

    assert box is not None
    assert box[0] >= 72


def test_coarse_clamp_still_allows_stable_thin_horizontal_leader():
    frames=[]
    for _ in range(7):
        f=np.full((180,420,3),105,np.uint8)
        _outlined_rect(f,55,91,88,96)  # thin 一 missed by coarse detector
        for x in (105,145,185,225):
            _outlined_rect(f,x,80,x+25,112)
        frames.append(f)

    slot=SlotPrior(
        y1=72,y2=120,expected_x_center=180,expected_height=48,
        expected_x1=100,expected_x2=255,
    )
    box=consensus_slot_bbox(frames,slot)

    assert box is not None
    assert box[0] <= 52


def test_coarse_guard_rejects_normal_glyph_outside_observed_fast_edge_support():
    frames=[]
    for _ in range(7):
        f=np.full((180,420,3),105,np.uint8)
        # A stable full-height component sits left of FAST's supported edge.
        # Without coarse support it may be scene structure and must not expand
        # the canonical bbox. Occlusion recovery uses the leftmost observed
        # FAST x1 instead of bypassing this guard.
        _outlined_rect(f,130,72,155,112)
        for x in (170,210,250,290):
            _outlined_rect(f,x,72,x+25,112)
        frames.append(f)

    slot=SlotPrior(
        y1=66,y2=118,expected_x_center=230,expected_height=52,
        expected_x1=160,expected_x2=320,
    )
    box=consensus_slot_bbox(frames,slot)

    assert box is not None
    assert box[0] >= 150
