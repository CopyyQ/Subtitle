from pathlib import Path
import json
import shutil
import subprocess
import cv2
import numpy as np
import pytest

from src.video_text.display_shape import draw_line_rectangles
from src.video_text.io import (
    CodecUnavailableError, encode_video, ffmpeg_video_codec,
    mux_audio, write_coordinate_json, write_srt,
)

def test_codec_mapping():
    assert ffmpeg_video_codec("h264")=="libx264"
    assert ffmpeg_video_codec("h265")=="libx265"
    with pytest.raises(ValueError):
        ffmpeg_video_codec("vp9")

def _work(name):
    p=Path(__file__).resolve().parent/".tmp_io"/name
    p.mkdir(parents=True,exist_ok=True)
    return p

def test_coordinate_json_roundtrip():
    p=_work("json")/"coords.json"
    write_coordinate_json(p,{"fps":30},[{"frame":1,"timestamp":1/30,"bbox":[1,2,3,4]}],[])
    d=json.loads(p.read_text())
    assert d["metadata"]["fps"]==30
    assert d["records"][0]["frame"]==1

def test_srt_writer_requires_real_text():
    p=_work("srt")/"a.srt"
    write_srt(p,[{"start":0.0,"end":1.0,"text":"你好"}])
    assert "你好" in p.read_text()

def test_h264_encode_roundtrip():
    tmp_path=_work("h264")
    frames=[np.zeros((64,96,3),np.uint8) for _ in range(5)]
    p=tmp_path/"h264.mp4"
    encode_video(frames,p,5.0,"h264")
    cap=cv2.VideoCapture(str(p))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==5
    cap.release()

def test_mux_does_not_truncate_video_when_source_audio_is_shorter():
    tmp_path=_work("mux")
    ff="/snap/bin/ffmpeg"
    video=tmp_path/"video.mp4"
    source=tmp_path/"source.mp4"
    out=tmp_path/"mux.mp4"
    subprocess.run([ff,"-y","-loglevel","error","-f","lavfi","-i","testsrc=size=96x64:rate=10:duration=2",
                    "-c:v","libx264","-pix_fmt","yuv420p",str(video)],check=True)
    subprocess.run([ff,"-y","-loglevel","error","-f","lavfi","-i","color=size=96x64:rate=10:duration=2",
                    "-f","lavfi","-i","sine=frequency=1000:duration=1",
                    "-map","0:v","-map","1:a","-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac",str(source)],check=True)
    mux_audio(video,source,out)
    cap=cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==20
    cap.release()

def test_coordinate_json_contains_every_processed_frame_with_empty_boxes():
    p=_work("json")/"coords_per_frame.json"
    write_coordinate_json(
        p,
        {"fps":30},
        [{"frame":1,"timestamp":1/30,"track_id":6,"subtitle_id":6,"line_id":0,
          "bbox":[1,2,30,40],"reconstructed":False}],
        [],
        frame_count=3,
        fps=30.0,
    )
    d=json.loads(p.read_text())
    assert len(d["frames"])==3
    assert d["frames"][0]["boxes"]==[]
    assert len(d["frames"][1]["boxes"])==1
    assert d["frames"][1]["boxes"][0]["bbox"]==[1,2,30,40]
    assert d["frames"][2]["boxes"]==[]


def test_line_rectangle_renderer_keeps_interior_unfilled():
    frame=np.zeros((80,120,3),np.uint8)
    draw_line_rectangles(
        frame,
        [{"bbox":[20,20,100,60],"track_id":1,"subtitle_id":1,"line_id":0}],
        thickness=2,
    )
    assert frame[40,60].tolist()==[0,0,0]
    assert frame[20,60].any()


def test_coordinate_json_can_store_display_polygons():
    p=_work("json")/"coords_with_shapes.json"
    shapes=[{
        "frame":10,
        "timestamp":0.4,
        "track_ids":[1,2],
        "polygon":[[100,100],[500,100],[500,150],[420,150],[420,206],[180,206],[180,150],[100,150]],
    }]
    write_coordinate_json(
        p,
        {"fps":25},
        [{"frame":10,"timestamp":0.4,"bbox":[100,100,500,206]}],
        [],
        display_shapes=shapes,
    )
    d=json.loads(p.read_text())
    assert d["display_shapes"]==shapes


def test_mux_handles_source_without_audio_without_changing_frame_count():
    tmp_path=_work("mux_no_audio")
    ff="/snap/bin/ffmpeg"
    video=tmp_path/"video.mp4"
    source=tmp_path/"source.mp4"
    out=tmp_path/"mux.mp4"
    subprocess.run([ff,"-y","-loglevel","error","-f","lavfi","-i",
                    "testsrc=size=96x64:rate=10:duration=1",
                    "-c:v","libx264","-pix_fmt","yuv420p",str(video)],check=True)
    shutil.copyfile(video,source)
    mux_audio(video,source,out)
    cap=cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==10
    cap.release()


def test_mux_longer_source_audio_does_not_add_video_frames():
    tmp_path=_work("mux_long_audio")
    ff="/snap/bin/ffmpeg"
    video=tmp_path/"video.mp4"
    source=tmp_path/"source.mp4"
    out=tmp_path/"mux.mp4"
    subprocess.run([ff,"-y","-loglevel","error","-f","lavfi","-i",
                    "testsrc=size=96x64:rate=10:duration=1",
                    "-c:v","libx264","-pix_fmt","yuv420p",str(video)],check=True)
    subprocess.run([ff,"-y","-loglevel","error","-f","lavfi","-i",
                    "color=size=96x64:rate=10:duration=1",
                    "-f","lavfi","-i","sine=frequency=1000:duration=3",
                    "-map","0:v","-map","1:a","-c:v","libx264",
                    "-pix_fmt","yuv420p","-c:a","aac",str(source)],check=True)
    mux_audio(video,source,out)
    cap=cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==10
    cap.release()
