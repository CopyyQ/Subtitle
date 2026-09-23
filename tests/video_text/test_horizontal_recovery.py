import pytest
from pathlib import Path
import cv2
import numpy as np

from src.video_text.types import Candidate
from src.video_text.horizontal_recovery import recover_horizontal_extent


def c(box):
    return Candidate(np.array(box,np.float32),.95,"HIGH")


def test_recovers_thin_leading_glyph_by_chaining_same_baseline_runs():
    frame=np.zeros((260,520,3),np.uint8)
    # Missing leading glyph: thin horizontal stroke.
    cv2.rectangle(frame,(40,118),(82,123),(255,255,255),-1)
    # Next characters.
    cv2.rectangle(frame,(92,100),(126,140),(255,255,255),-1)
    cv2.rectangle(frame,(136,100),(170,140),(255,255,255),-1)
    cv2.rectangle(frame,(180,100),(214,140),(255,255,255),-1)
    candidate=c([108,94,220,146])
    out=recover_horizontal_extent(frame,candidate)
    assert out.bbox[0] <= 40
    assert out.bbox[2] >= 214
    assert out.bbox[1] == candidate.bbox[1]
    assert out.bbox[3] == candidate.bbox[3]


def test_far_white_object_is_not_attached():
    frame=np.zeros((260,520,3),np.uint8)
    cv2.rectangle(frame,(20,105),(55,140),(255,255,255),-1)
    cv2.rectangle(frame,(140,100),(180,140),(255,255,255),-1)
    cv2.rectangle(frame,(190,100),(230,140),(255,255,255),-1)
    candidate=c([135,94,235,146])
    out=recover_horizontal_extent(frame,candidate)
    assert out.bbox[0] >= 130


def test_low_candidate_is_not_expanded():
    frame=np.zeros((260,520,3),np.uint8)
    cv2.rectangle(frame,(40,118),(82,123),(255,255,255),-1)
    cv2.rectangle(frame,(92,100),(126,140),(255,255,255),-1)
    candidate=Candidate(np.array([108,94,220,146],np.float32),.62,"LOW")
    out=recover_horizontal_extent(frame,candidate)
    assert np.allclose(out.bbox,candidate.bbox)


def test_real_frame_1648_recovers_leading_yi_glyph():
    root=Path(__file__).resolve().parents[2]
    if not (root/"AI Engineer test.mp4").exists():
        pytest.skip("private evaluation video is not included in the Git repository")
    cap=cv2.VideoCapture(str(root/"AI Engineer test.mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES,1648)
    ok,frame=cap.read()
    cap.release()
    assert ok
    raw=c([137.85884,844.7946,620.2933,903.3107])
    out=recover_horizontal_extent(frame,raw)
    # The visible 一 component occupies about x=98..143.
    assert out.bbox[0] <= 100
    assert out.bbox[2] >= 620

def test_frame_horizontal_recovery_expands_high_but_preserves_low():
    from src.video_text.types import FrameDetections
    from src.video_text.horizontal_recovery import recover_frame_horizontal_extents
    frame=np.zeros((260,520,3),np.uint8)
    cv2.rectangle(frame,(40,118),(82,123),(255,255,255),-1)
    cv2.rectangle(frame,(92,100),(126,140),(255,255,255),-1)
    high=c([108,94,220,146])
    low=Candidate(np.array([300,94,360,146],np.float32),.62,"LOW")
    fd=FrameDetections(3,.1,[high],[low])
    out=recover_frame_horizontal_extents(frame,fd)
    assert out.high[0].bbox[0] <= 40
    assert np.allclose(out.low[0].bbox,low.bbox)

def test_tall_white_background_run_is_not_attached():
    frame=np.zeros((260,520,3),np.uint8)
    # Candidate text.
    cv2.rectangle(frame,(140,100),(180,140),(255,255,255),-1)
    cv2.rectangle(frame,(190,100),(230,140),(255,255,255),-1)
    # Near-left bright background spans much more vertically than the font.
    cv2.rectangle(frame,(90,75),(128,158),(255,255,255),-1)
    candidate=c([135,94,235,146])
    out=recover_horizontal_extent(frame,candidate)
    assert out.bbox[0] >= 130


def test_off_baseline_thin_run_is_not_attached():
    frame=np.zeros((260,520,3),np.uint8)
    cv2.rectangle(frame,(140,100),(180,140),(255,255,255),-1)
    cv2.rectangle(frame,(190,100),(230,140),(255,255,255),-1)
    # Thin run is horizontally close but vertically near the bottom edge,
    # unlike a leading 一 centered on the subtitle baseline.
    cv2.rectangle(frame,(95,145),(128,150),(255,255,255),-1)
    candidate=c([135,94,235,146])
    out=recover_horizontal_extent(frame,candidate)
    assert out.bbox[0] >= 130


def test_horizontal_recovery_does_not_chain_beyond_one_line_height():
    frame=np.zeros((260,520,3),np.uint8)
    cv2.rectangle(frame,(140,100),(180,140),(255,255,255),-1)
    cv2.rectangle(frame,(190,100),(230,140),(255,255,255),-1)
    # Several plausible glyph-like runs form a chain far to the left.
    for x in (100,60,20):
        cv2.rectangle(frame,(x,105),(x+30,135),(255,255,255),-1)
    candidate=c([135,94,235,146])
    out=recover_horizontal_extent(frame,candidate)
    assert out.bbox[0] >= 80


def test_track_level_horizontal_recovery_expands_confirmed_high_observations(tmp_path):
    from src.video_text.types import SubtitleTrack, TrackObservation
    from src.video_text.horizontal_recovery import recover_tracks_horizontal_extents

    video=tmp_path/"track_recovery.mp4"
    writer=cv2.VideoWriter(
        str(video),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,160)
    )
    assert writer.isOpened()
    for _ in range(3):
        fr=np.zeros((160,320,3),np.uint8)
        cv2.rectangle(fr,(40,122),(78,126),(255,255,255),-1)
        cv2.rectangle(fr,(88,105),(120,142),(255,255,255),-1)
        cv2.rectangle(fr,(130,105),(162,142),(255,255,255),-1)
        cv2.rectangle(fr,(172,105),(204,142),(255,255,255),-1)
        writer.write(fr)
    writer.release()

    t=SubtitleTrack(1,confirmed=True)
    for f in range(3):
        t.observations[f]=TrackObservation(
            f,np.array([108,94,220,146],np.float32),.95,"HIGH"
        )

    out,count=recover_tracks_horizontal_extents(video,[t])
    assert count==3
    assert all(out[0].observations[f].bbox[0] <= 40 for f in range(3))
