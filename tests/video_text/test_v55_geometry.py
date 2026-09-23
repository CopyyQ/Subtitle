import pytest
import cv2
import numpy as np

from src.video_text.v55_geometry import (
    enforce_non_overlapping_lines,
    line_overlap_area,
)


def _outlined_line(frame, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (10,10,10), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (245,245,245), -1)


def test_two_line_boxes_never_overlap_and_may_touch_at_seam():
    frame=np.full((240,420,3),100,np.uint8)
    _outlined_line(frame,50,90,360,125)
    _outlined_line(frame,150,132,270,166)
    boxes=[
        np.array([45,86,365,137],np.float32),
        np.array([145,120,275,171],np.float32),
    ]
    top,bottom=enforce_non_overlapping_lines(frame,boxes)
    assert top[3] <= bottom[1]
    assert line_overlap_area(top,bottom) == 0.0
    assert top[3] >= 127
    assert bottom[1] <= 130


def test_non_overlapping_input_is_not_expanded_to_touch():
    frame=np.full((240,420,3),100,np.uint8)
    boxes=[
        np.array([45,86,365,126],np.float32),
        np.array([145,138,275,172],np.float32),
    ]
    out=enforce_non_overlapping_lines(frame,boxes)
    assert np.allclose(out[0],boxes[0])
    assert np.allclose(out[1],boxes[1])


def test_short_centered_lower_line_stays_independent_from_long_upper_line():
    frame=np.full((240,420,3),100,np.uint8)
    boxes=[
        np.array([30,82,390,130],np.float32),
        np.array([180,124,240,170],np.float32),
    ]
    top,bottom=enforce_non_overlapping_lines(frame,boxes)
    assert top[0] == 30 and top[2] == 390
    assert bottom[0] == 180 and bottom[2] == 240
    assert top[3] <= bottom[1]


def test_three_ordered_boxes_are_pairwise_non_overlapping():
    frame=np.full((300,420,3),100,np.uint8)
    boxes=[
        np.array([40,70,380,130],np.float32),
        np.array([90,118,330,180],np.float32),
        np.array([160,168,260,225],np.float32),
    ]
    out=enforce_non_overlapping_lines(frame,boxes)
    assert all(out[i][3] <= out[i+1][1] for i in range(len(out)-1))


def test_refinement_shrinks_large_anchor_to_visible_outline_plus_small_pad():
    from src.video_text.v55_geometry import refine_line_bbox
    from src.video_text.v5_weak_birth import SlotPrior
    frame=np.full((220,420,3),100,np.uint8)
    _outlined_line(frame,100,90,320,128)
    anchor=np.array([70,80,350,140],np.float32)
    slot=SlotPrior(75,145,210,70,expected_x1=70,expected_x2=350)
    result=refine_line_bbox(frame,anchor,slot,pad_px=2)
    x1,y1,x2,y2=result.bbox
    assert 95 <= x1 <= 100
    assert 320 <= x2 <= 325
    assert 85 <= y1 <= 90
    assert 128 <= y2 <= 133
    assert result.used_anchor is False


def test_one_frame_occlusion_falls_back_near_anchor_instead_of_collapsing():
    from src.video_text.v55_geometry import refine_line_bbox
    from src.video_text.v5_weak_birth import SlotPrior
    frame=np.full((220,420,3),100,np.uint8)
    anchor=np.array([98,86,322,132],np.float32)
    slot=SlotPrior(80,138,210,58,expected_x1=98,expected_x2=322)
    result=refine_line_bbox(frame,anchor,slot)
    assert result.used_anchor is True
    assert np.allclose(result.bbox,anchor)


def test_smoother_allows_slow_two_pixel_motion_but_rejects_one_frame_spike():
    from src.video_text.v55_geometry import LineGeometry, smooth_line_geometry
    anchor=np.array([100,90,320,130],np.float32)
    seq=[]
    for dx in [0,1,2,20,3,4,5]:
        seq.append(LineGeometry(
            anchor+np.array([dx,0,dx,0],np.float32),
            1.0,
            False,
        ))
    out=smooth_line_geometry(seq,anchor,window=3,max_step_px=2.0)
    assert max(
        abs(float(out[i][0]-out[i-1][0]))
        for i in range(1,len(out))
    ) <= 2.01
    assert out[3][0] < 110
    assert out[-1][0] > out[0][0]


def test_refinement_does_not_grow_far_outside_anchor_from_scene_noise():
    from src.video_text.v55_geometry import refine_line_bbox
    from src.video_text.v5_weak_birth import SlotPrior

    frame=np.full((220,420,3),100,np.uint8)
    _outlined_line(frame,100,92,320,128)
    # Scene-like white/dark fragments just outside the trusted temporal anchor.
    _outlined_line(frame,165,80,215,85)
    _outlined_line(frame,190,136,240,141)

    anchor=np.array([96,88,324,132],np.float32)
    slot=SlotPrior(88,132,210,44,expected_x1=96,expected_x2=324)
    result=refine_line_bbox(frame,anchor,slot,pad_px=2)

    assert result.bbox[1] >= anchor[1]-2.01
    assert result.bbox[3] <= anchor[3]+2.01


def test_production_scene_noise_does_not_expand_far_beyond_temporal_anchor():
    from pathlib import Path
    from src.video_text.v55_geometry import refine_line_bbox
    from src.video_text.v5_weak_birth import SlotPrior

    root=Path(__file__).resolve().parents[2]
    if not (root/"AI Engineer test.mp4").exists():
        pytest.skip("private evaluation video is not included in the Git repository")
    cap=cv2.VideoCapture(str(root/"AI Engineer test.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES,3151)
    ok,frame=cap.read()
    cap.release()
    assert ok

    anchor=np.array([161,840,559,909],np.float32)
    slot=SlotPrior(
        840,909,361.4590759277344,69.0,
        expected_x1=162.40557861328125,
        expected_x2=558.0595703125,
    )
    result=refine_line_bbox(frame,anchor,slot,pad_px=3)

    assert result.bbox[0] >= anchor[0]-3.01
    assert result.bbox[2] <= anchor[2]+3.01
    assert result.bbox[1] >= anchor[1]-1e-3
    assert result.bbox[3] <= anchor[3]+1e-3
