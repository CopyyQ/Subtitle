from pathlib import Path
import json

import cv2
import numpy as np

from src.video_text.pipeline import PipelineConfig, SubtitlePipeline
from src.video_text.types import Candidate


ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/"tests/video_text/.tmp_v55_pipeline"


class FakeBackend:
    batch_size=4
    def detect_batch(self,images):
        out=[]
        for _ in images:
            out.append(([
                Candidate(np.array([58,10,270,58],np.float32),.96,"HIGH","fake"),
                Candidate(np.array([138,63,182,98],np.float32),.94,"HIGH","fake"),
            ],[]))
        return out


def _outlined_rect(frame,x1,y1,x2,y2):
    cv2.rectangle(frame,(x1-3,y1-3),(x2+3,y2+3),(10,10,10),-1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(245,245,245),-1)


def _make_video(path,n=10):
    path.parent.mkdir(parents=True,exist_ok=True)
    w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,220))
    assert w.isOpened()
    for _ in range(n):
        f=np.full((220,320,3),100,np.uint8)
        for x in (65,105,145,185,225):
            _outlined_rect(f,x,142,x+22,175)
        _outlined_rect(f,145,183,175,214)
        w.write(f)
    w.release()


def test_pipeline_v55_emits_two_independent_line_rectangles():
    src=WORK/"two_line.mp4"
    _make_video(src)
    out=WORK/"v55.mp4"
    cfg=PipelineConfig(
        temporal_mode="v5_5",
        validate_chinese=False,
        roi_bottom_fraction=.45,
        output_codec="h264",
    )
    result=SubtitlePipeline(cfg,backend=FakeBackend()).run(
        src,out,max_frames=10
    )
    data=json.loads(result.coordinate_json.read_text())
    assert data["metadata"]["temporal_mode"]=="v5_5"

    by_frame={}
    for row in data["records"]:
        by_frame.setdefault(row["frame"],[]).append(row)
    assert all(len(by_frame[f])==2 for f in range(10))
    for rows in by_frame.values():
        rows=sorted(rows,key=lambda r:r["line_id"])
        assert {r["subtitle_id"] for r in rows}=={rows[0]["subtitle_id"]}
        assert [r["line_id"] for r in rows]==[0,1]
        assert rows[0]["bbox"][3] <= rows[1]["bbox"][1]

    shapes=[s for s in data["display_shapes"] if s["frame"]==5]
    assert len(shapes)==2
    assert all(len(s["polygon"])==4 for s in shapes)

    cap=cv2.VideoCapture(str(out))
    assert cap.isOpened()
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT))==10
    assert cap.get(cv2.CAP_PROP_FPS)==10.0
    assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))==320
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))==220
    cap.release()


def test_v55_signature_differs_from_v5_for_same_source():
    src=WORK/"signature.mp4"
    _make_video(src,n=1)
    p5=SubtitlePipeline(PipelineConfig(
        temporal_mode="v5",validate_chinese=False
    ),backend=FakeBackend())
    p55=SubtitlePipeline(PipelineConfig(
        temporal_mode="v5_5",validate_chinese=False
    ),backend=FakeBackend())
    s5=p5._signature(src,1)
    s55=p55._signature(src,1)
    assert s5!=s55
    assert s5["temporal_mode"]=="v5"
    assert s55["temporal_mode"]=="v5_5"


def test_weak_validation_uses_separated_bbox_not_overlapping_strong_text():
    from src.video_text.types import SubtitleTrack, TrackObservation

    src=WORK/"weak_validation_order.mp4"
    w=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,220))
    assert w.isOpened()
    frame=np.full((220,320,3),100,np.uint8)
    _outlined_rect(frame,80,90,240,120)
    w.write(frame)
    w.release()

    strong=SubtitleTrack(1,confirmed=True)
    strong.observations[0]=TrackObservation(
        0,np.array([76,85,244,130],np.float32),.95,"V55_LINE"
    )
    weak=SubtitleTrack(2,confirmed=True)
    weak.observations[0]=TrackObservation(
        0,np.array([120,110,200,150],np.float32),None,"V5_WEAK"
    )
    identity={1:(7,0),2:(7,1)}

    class HeightSensitiveRecognizer:
        def recognize(self,crop):
            # Raw weak box is 40 px tall and overlaps the real strong text.
            # A correctly separated validation crop is much shorter and empty.
            if crop.shape[0] >= 30:
                return "中",.99
            return "",0.0

    pipe=SubtitlePipeline(PipelineConfig(
        temporal_mode="v5_5",validate_chinese=True
    ))
    kept,rejected=pipe._validate_v55_weak_post_separation(
        src,[strong],[weak],identity,HeightSensitiveRecognizer()
    )
    assert kept==[]
    assert rejected==1
