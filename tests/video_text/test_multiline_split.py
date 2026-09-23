import cv2
import numpy as np

from src.video_text.types import Candidate
from src.video_text.line_split import split_candidate_by_projection
from src.video_text.association import AssociationConfig, compatible


def make_frame(two_lines=True):
    frame=np.zeros((300,640,3),np.uint8)
    # top subtitle line, intentionally wide
    for x in range(100,501,45):
        cv2.rectangle(frame,(x,100),(x+26,136),(255,255,255),-1)
    if two_lines:
        for x in range(260,351,45):
            cv2.rectangle(frame,(x,154),(x+26,190),(255,255,255),-1)
    return frame


def test_projection_splits_two_lines_even_when_detector_box_only_covers_top_line():
    frame=make_frame(two_lines=True)
    candidate=Candidate(np.array([90,94,520,142],np.float32),.96,"HIGH")
    out=split_candidate_by_projection(frame,candidate)
    assert len(out)==2
    assert all(x.level=="HIGH" for x in out)
    assert out[0].bbox[1] < 110
    assert out[0].bbox[3] > 130
    assert out[1].bbox[1] < 160
    assert out[1].bbox[3] > 185
    assert out[1].bbox[0] > out[0].bbox[0]
    assert out[1].bbox[2] < out[0].bbox[2]


def test_projection_keeps_single_line_as_single_candidate():
    frame=make_frame(two_lines=False)
    candidate=Candidate(np.array([90,94,520,142],np.float32),.96,"HIGH")
    out=split_candidate_by_projection(frame,candidate)
    assert len(out)==1
    assert np.allclose(out[0].bbox,candidate.bbox)


def test_association_rejects_different_subtitle_baselines():
    cfg=AssociationConfig()
    top=np.array([90,100,520,142],np.float32)
    lower=np.array([250,154,360,194],np.float32)
    same=np.array([92,101,522,143],np.float32)
    assert compatible(top,same,cfg)
    assert not compatible(top,lower,cfg)

def test_frame_detections_are_split_and_regrouped_per_line():
    from src.video_text.types import FrameDetections
    from src.video_text.line_split import split_frame_detections_by_projection
    frame=make_frame(two_lines=True)
    fd=FrameDetections(
        frame_index=12,
        timestamp=.4,
        high=[Candidate(np.array([90,94,520,142],np.float32),.96,"HIGH")],
        low=[],
    )
    out=split_frame_detections_by_projection(frame,fd)
    assert out.frame_index==12
    assert len(out.high)==2
    assert len(out.low)==0
    assert out.high[0].bbox[3] < out.high[1].bbox[1]

def test_low_candidate_is_never_expanded_into_new_lines():
    frame=make_frame(two_lines=True)
    candidate=Candidate(np.array([90,94,520,142],np.float32),.62,"LOW")
    out=split_candidate_by_projection(frame,candidate)
    assert len(out)==1
    assert np.allclose(out[0].bbox,candidate.bbox)


def test_off_center_white_region_below_is_not_a_subtitle_line():
    frame=np.zeros((300,640,3),np.uint8)
    for x in range(100,501,45):
        cv2.rectangle(frame,(x,100),(x+26,136),(255,255,255),-1)
    # Same-height white object, but far to the right instead of centered.
    for x in range(470,561,45):
        cv2.rectangle(frame,(x,154),(x+26,190),(255,255,255),-1)
    candidate=Candidate(np.array([90,94,520,142],np.float32),.96,"HIGH")
    out=split_candidate_by_projection(frame,candidate)
    assert len(out)==1


def test_tiny_secondary_run_is_not_treated_as_second_subtitle_line():
    frame=np.zeros((300,640,3),np.uint8)
    for x in range(100,501,45):
        cv2.rectangle(frame,(x,100),(x+26,136),(255,255,255),-1)
    # Centered but much shorter than the subtitle font height.
    cv2.rectangle(frame,(315,154),(340,162),(255,255,255),-1)
    candidate=Candidate(np.array([90,94,520,142],np.float32),.96,"HIGH")
    out=split_candidate_by_projection(frame,candidate)
    assert len(out)==1
