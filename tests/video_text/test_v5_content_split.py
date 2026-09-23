from pathlib import Path

import cv2
import numpy as np

from src.video_text.types import SubtitleTrack, TrackObservation
from src.video_text.v5_content_split import split_tracks_on_text_change


WORK=Path(__file__).resolve().parent/".tmp_v5_content_split"


def _outlined_rect(frame,x1,y1,x2,y2):
    cv2.rectangle(frame,(x1-3,y1-3),(x2+3,y2+3),(20,20,20),-1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(215,215,215),-1)


def _caption_a():
    f=np.full((180,420,3),105,np.uint8)
    for x in (90,130,170,210):
        _outlined_rect(f,x,72,x+25,112)
    return f


def _caption_b():
    f=np.full((180,420,3),105,np.uint8)
    for x in (135,175,215,255,295):
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
            f,np.array([70,66,335,118],np.float32),.95,"HIGH"
        )
    return t


def test_split_tracks_on_text_change_separates_two_stable_captions():
    frames=[_caption_a() for _ in range(10)] + [_caption_b() for _ in range(10)]
    src=WORK/"two_stable.mp4"
    _write_video(src,frames)

    out=split_tracks_on_text_change(src,[_track(len(frames))])

    assert len(out)==2
    assert out[0].sorted_frames()==list(range(10))
    assert out[1].sorted_frames()==list(range(10,20))
    assert out[1].content_boundary_before is True


def test_split_tracks_on_text_change_ignores_one_frame_occlusion():
    frames=[_caption_a() for _ in range(20)]
    cv2.rectangle(frames[9],(70,60),(185,125),(80,80,80),-1)
    src=WORK/"one_frame_occlusion.mp4"
    _write_video(src,frames)

    out=split_tracks_on_text_change(src,[_track(len(frames))])

    assert len(out)==1
    assert out[0].sorted_frames()==list(range(20))
