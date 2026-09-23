import numpy as np
from src.video_text.types import SubtitleTrack, TrackObservation
from src.video_text.smoothing import smooth_track

def make_track(boxes,tid=1):
    t=SubtitleTrack(tid,confirmed=True)
    for i,b in enumerate(boxes):
        t.observations[i]=TrackObservation(i,np.array(b,np.float32),.9,"HIGH")
    return t

def test_centered_median_reduces_single_frame_jitter():
    t=make_track([
        [100,800,400,850],[101,800,401,850],[108,806,408,856],
        [101,800,401,850],[100,800,400,850],
    ])
    out=smooth_track(t,window=5)
    assert out.observations[2].bbox[0] <= 101
    assert out.observations[2].bbox[1] == 800

def test_short_track_is_preserved():
    t=make_track([[100,800,400,850],[110,805,410,855]])
    out=smooth_track(t,window=5)
    assert np.allclose(out.observations[0].bbox,t.observations[0].bbox)
    assert np.allclose(out.observations[1].bbox,t.observations[1].bbox)

def test_tracks_are_smoothed_independently():
    a=smooth_track(make_track([[100,800,400,850]]*3,1),window=3)
    b=smooth_track(make_track([[500,900,650,940]]*3,2),window=3)
    assert a.observations[1].bbox[0]==100
    assert b.observations[1].bbox[0]==500
