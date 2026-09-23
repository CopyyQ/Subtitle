from pathlib import Path
import json

import cv2
import numpy as np

from src.video_text.pipeline import PipelineConfig, SubtitlePipeline
from src.video_text.types import Candidate


ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/"tests/video_text/.tmp_v5_pipeline"


class FakeBackend:
    batch_size=4
    def __init__(self,n):
        self.n=n
        self.position=0
    def detect_batch(self,images):
        out=[]
        for _ in images:
            # Deliberately only a top-line coarse box. V5 must discover the
            # second row from the first bootstrap frames.
            out.append(([
                Candidate(
                    np.array([58,8,270,52],np.float32),
                    .96,"HIGH","fake"
                )
            ],[]))
            self.position+=1
        return out


def _outlined_rect(frame,x1,y1,x2,y2):
    cv2.rectangle(frame,(x1-3,y1-3),(x2+3,y2+3),(10,10,10),-1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(245,245,245),-1)


def _make_video(path,n=10):
    path.parent.mkdir(parents=True,exist_ok=True)
    w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,200))
    assert w.isOpened()
    for _ in range(n):
        f=np.full((200,320,3),100,np.uint8)
        for x in range(65,246,36):
            _outlined_rect(f,x,135,x+22,163)
        _outlined_rect(f,145,168,175,194)
        w.write(f)
    w.release()


def test_pipeline_v5_locks_two_line_slots_without_per_frame_split():
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"two_line.mp4"
    _make_video(src)
    out=WORK/"v5.mp4"

    cfg=PipelineConfig(
        temporal_mode="v5",
        validate_chinese=False,
        roi_bottom_fraction=.45,
        output_codec="h264",
    )
    result=SubtitlePipeline(cfg,backend=FakeBackend(10)).run(
        src,out,max_frames=10
    )
    data=json.loads(result.coordinate_json.read_text())

    assert data["metadata"]["temporal_mode"]=="v5"
    assert data["metadata"]["v5_segment_count"]==1
    assert data["metadata"]["v5_two_line_segment_count"]==1
    assert data["metadata"]["v5_strong_track_count"]==2
    assert data["metadata"]["stabilized_corner_motion_p95_px"]==0.0

    by_frame={}
    for row in data["records"]:
        by_frame.setdefault(row["frame"],[]).append(row)
    assert all(len(by_frame[f])==2 for f in range(10))

    shapes=[x for x in data["display_shapes"] if x["frame"]==5]
    assert len(shapes)==1
    assert len(shapes[0]["track_ids"])==2
    assert len(shapes[0]["polygon"])>4


class EmptyRecognizer:
    def recognize(self,crop):
        return "",0.0


def test_pipeline_v5_rejects_unreadable_pixel_only_second_slot():
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"pixel_only_second_slot.mp4"
    _make_video(src)
    out=WORK/"v5_pixel_only_filtered.mp4"

    cfg=PipelineConfig(
        temporal_mode="v5",
        validate_chinese=True,
        roi_bottom_fraction=.45,
        output_codec="h264",
    )
    result=SubtitlePipeline(
        cfg,
        backend=FakeBackend(10),
        recognizer=EmptyRecognizer(),
    ).run(src,out,max_frames=10)
    data=json.loads(result.coordinate_json.read_text())

    # Top slot is backed by FAST and stays retained_uncertain. The second slot
    # exists only because image bootstrap proposed it; with no repeated CJK
    # evidence it must be removed rather than become a false lower bbox.
    assert data["metadata"]["v5_pixel_only_slot_count"] >= 1
    by_frame={}
    for row in data["records"]:
        by_frame.setdefault(row["frame"],[]).append(row)
    assert all(len(by_frame[f])==1 for f in range(10))
