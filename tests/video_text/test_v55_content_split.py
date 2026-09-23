from pathlib import Path

import cv2
import numpy as np

from src.video_text.types import SubtitleTrack, TrackObservation
from src.video_text.v55_content_split import split_tracks_on_text_change_windowed


WORK=Path(__file__).resolve().parent/".tmp_v55_content_split"


def _outlined_rect(frame,x1,y1,x2,y2):
    cv2.rectangle(frame,(x1-3,y1-3),(x2+3,y2+3),(20,20,20),-1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(220,220,220),-1)


def _caption(xs):
    f=np.full((180,420,3),105,np.uint8)
    for x in xs:
        _outlined_rect(f,x,72,x+25,112)
    return f


def _write_video(path,frames):
    path.parent.mkdir(parents=True,exist_ok=True)
    w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(420,180))
    assert w.isOpened()
    for frame in frames:
        w.write(frame)
    w.release()


def _track(n):
    t=SubtitleTrack(1,confirmed=True)
    for f in range(n):
        t.observations[f]=TrackObservation(
            f,np.array([65,64,350,120],np.float32),.95,"HIGH"
        )
    return t


def test_windowed_split_separates_two_stable_captions():
    a=_caption((90,130,170,210))
    b=_caption((135,175,215,255,295))
    frames=[a.copy() for _ in range(10)]+[b.copy() for _ in range(10)]
    src=WORK/"stable_change.mp4"
    _write_video(src,frames)
    out=split_tracks_on_text_change_windowed(src,[_track(20)],window=3)
    assert len(out)==2
    assert out[0].sorted_frames()==list(range(10))
    assert out[1].sorted_frames()==list(range(10,20))


def test_windowed_split_ignores_one_frame_occlusion():
    a=_caption((90,130,170,210))
    frames=[a.copy() for _ in range(20)]
    cv2.rectangle(frames[9],(65,60),(190,125),(80,80,80),-1)
    src=WORK/"one_frame_occlusion.mp4"
    _write_video(src,frames)
    out=split_tracks_on_text_change_windowed(src,[_track(20)],window=3)
    assert len(out)==1


def test_similar_width_internal_pattern_change_is_detected():
    a=_caption((85,130,175,220,265))
    b=_caption((85,115,175,235,265))
    frames=[a.copy() for _ in range(9)]+[b.copy() for _ in range(9)]
    src=WORK/"similar_shape.mp4"
    _write_video(src,frames)
    out=split_tracks_on_text_change_windowed(src,[_track(18)],window=3)
    assert len(out)==2
    assert out[0].sorted_frames()[-1] in (8,9)


def test_two_frame_crossfade_makes_one_boundary_not_extra_fragments():
    a=_caption((90,130,170,210))
    b=_caption((135,175,215,255,295))
    blend1=cv2.addWeighted(a,.67,b,.33,0)
    blend2=cv2.addWeighted(a,.33,b,.67,0)
    frames=[a.copy() for _ in range(8)]+[blend1,blend2]+[b.copy() for _ in range(8)]
    src=WORK/"crossfade.mp4"
    _write_video(src,frames)
    out=split_tracks_on_text_change_windowed(src,[_track(18)],window=3)
    assert len(out)==2
    boundary=out[1].sorted_frames()[0]
    assert 8 <= boundary <= 11
