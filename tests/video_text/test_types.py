import numpy as np
from src.video_text.types import Candidate, FrameDetections, TrackObservation, SubtitleTrack

def test_candidate_normalizes_bbox_to_float32():
    c = Candidate(np.array([1, 2, 3, 4]), 0.91, "HIGH")
    assert c.bbox.dtype == np.float32
    assert c.level == "HIGH"

def test_track_returns_frames_in_order():
    t = SubtitleTrack(track_id=7)
    t.observations[9] = TrackObservation(9, np.array([1,2,3,4]), .9, "HIGH")
    t.observations[3] = TrackObservation(3, np.array([1,2,3,4]), .8, "LOW")
    assert t.sorted_frames() == [3, 9]
