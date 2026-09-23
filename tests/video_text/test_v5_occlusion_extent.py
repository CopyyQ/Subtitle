import cv2
import numpy as np

from src.video_text.v5_extent import consensus_slot_bbox
from src.video_text.v5_weak_birth import SlotPrior


def _outlined_rect(frame,x1,y1,x2,y2):
    cv2.rectangle(frame,(x1-3,y1-3),(x2+3,y2+3),(20,20,20),-1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(210,210,210),-1)


def test_occluded_left_glyph_is_recovered_from_few_clear_frames_without_right_noise_expansion():
    frames=[]
    for i in range(9):
        f=np.full((180,420,3),105,np.uint8)
        # Stable main text.
        for x in (170,210,250,290):
            _outlined_rect(f,x,72,x+25,112)

        # The left glyph is hidden in most frames, but becomes clearly visible
        # in two frames. This mimics the hoe handle crossing the glyph.
        if i in (4,5):
            _outlined_rect(f,130,72,155,112)

        # Small bright outlined detail on the right survives in two frames,
        # but it is too small to be a real glyph and must not stretch bbox.
        if i in (6,7):
            _outlined_rect(f,340,90,347,101)

        frames.append(f)

    slot=SlotPrior(
        y1=66,y2=118,
        expected_x_center=235,
        expected_height=52,
    )
    box=consensus_slot_bbox(frames,slot)

    assert box is not None
    # Recover the temporarily occluded left glyph (+ outline padding).
    assert box[0] <= 127
    # Do not absorb the tiny right-side nuisance.
    assert box[2] < 335
    assert box[1] == 66
    assert box[3] == 118


def test_occlusion_recovery_rejects_partial_width_right_artifact_seen_twice():
    frames=[]
    for i in range(9):
        f=np.full((180,420,3),105,np.uint8)
        for x in (170,210,250,290):
            _outlined_rect(f,x,72,x+25,112)
        if i in (4,5):
            _outlined_rect(f,130,72,155,112)
        # Wider than a speck but still much narrower than the real glyphs.
        # This models the persistent bright handle/background detail around
        # x~590 in the 1:43 segment.
        if i in (6,7):
            _outlined_rect(f,340,78,355,108)
        frames.append(f)

    slot=SlotPrior(y1=66,y2=118,expected_x_center=235,expected_height=52)
    box=consensus_slot_bbox(frames,slot)

    assert box is not None
    assert box[0] <= 127
    assert box[2] < 335
