from pathlib import Path
import json
import cv2
import numpy as np
from src.video_text.types import Candidate
from src.video_text.pipeline import PipelineConfig, SubtitlePipeline

ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/"tests/video_text/.tmp_pipeline"

class FakeBackend:
    batch_size=4
    def __init__(self,timeline):
        self.timeline=timeline
        self.position=0
        self.calls=0
    def detect_batch(self,images):
        self.calls+=1
        out=[]
        for _ in images:
            high,low=self.timeline.get(self.position,([],[]))
            out.append((high,low))
            self.position+=1
        return out

def c(b,score=.95,level="HIGH"):
    return Candidate(np.array(b,np.float32),score,level,"fake")

def make_video(path,n=13):
    path.parent.mkdir(parents=True,exist_ok=True)
    w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(120,80))
    assert w.isOpened()
    for i in range(n):
        fr=np.full((80,120,3),i,np.uint8)
        w.write(fr)
    w.release()

def test_pipeline_recovers_internal_gap_but_not_true_end_and_reuses_cache():
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"synthetic.mp4"
    make_video(src)
    box=[20,4,95,20]
    timeline={}
    for i in list(range(0,5))+list(range(6,10)):
        timeline[i]=([c(box)],[])
    backend=FakeBackend(timeline)
    cfg=PipelineConfig(validate_chinese=False,roi_bottom_fraction=.45,
                       max_internal_gap=2,smoothing_window=5,output_codec="h264")
    p=SubtitlePipeline(cfg,backend=backend)
    out1=WORK/"out1.mp4"
    r1=p.run(src,out1,max_frames=13)
    first_calls=backend.calls
    assert first_calls>0
    data=json.loads(r1.coordinate_json.read_text())
    frames={r["frame"] for r in data["records"]}
    assert 5 in frames
    assert 10 not in frames and 11 not in frames and 12 not in frames
    event_types=[e["event_type"] for e in data["events"]]
    assert "internal_miss_recovered" in event_types
    assert "track_end" in event_types

    # Same source/config/output directory must reuse the detection cache.
    backend.position=0
    out2=WORK/"out2.mp4"
    p.run(src,out2,max_frames=13)
    assert backend.calls==first_calls


def test_pipeline_recovers_leading_glyph_extent_before_tracking():
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"horizontal_recovery.mp4"
    writer=cv2.VideoWriter(
        str(src),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,160)
    )
    assert writer.isOpened()
    for _ in range(5):
        fr=np.zeros((160,320,3),np.uint8)
        # Leading thin glyph, then body glyphs.
        cv2.rectangle(fr,(40,122),(78,126),(255,255,255),-1)
        cv2.rectangle(fr,(88,105),(120,142),(255,255,255),-1)
        cv2.rectangle(fr,(130,105),(162,142),(255,255,255),-1)
        cv2.rectangle(fr,(172,105),(204,142),(255,255,255),-1)
        writer.write(fr)
    writer.release()

    # ROI starts at y=88; detector intentionally clips the leading glyph.
    timeline={
        i:([c([105,15,215,60])],[])
        for i in range(5)
    }
    backend=FakeBackend(timeline)
    cfg=PipelineConfig(
        validate_chinese=False,
        roi_bottom_fraction=.45,
        max_internal_gap=2,
        smoothing_window=5,
        output_codec="h264",
    )
    out=WORK/"horizontal_recovery_out.mp4"
    result=SubtitlePipeline(cfg,backend=backend).run(src,out,max_frames=5)
    data=json.loads(result.coordinate_json.read_text())
    assert data["records"]
    assert min(r["bbox"][0] for r in data["records"]) <= 40


def test_pipeline_does_not_bridge_when_actual_same_line_detection_already_exists():
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"covered_gap.mp4"
    make_video(src,n=3)

    # Frame 0/2: small track. Frame 1: a wider real detection covers the
    # same baseline. Width ratio is deliberately > association gate so it
    # forms a separate actual track rather than continuing the small one.
    timeline={
        0:([c([50,10,110,30])],[]),
        1:([c([10,10,110,30])],[]),
        2:([c([50,10,110,30])],[]),
    }
    backend=FakeBackend(timeline)
    cfg=PipelineConfig(
        validate_chinese=False,
        roi_bottom_fraction=.45,
        max_internal_gap=2,
        smoothing_window=5,
        output_codec="h264",
    )
    out=WORK/"covered_gap_out.mp4"
    result=SubtitlePipeline(cfg,backend=backend).run(src,out,max_frames=3)
    data=json.loads(result.coordinate_json.read_text())

    assert data["metadata"]["suppressed_reconstructed_overlap_count"]==1
    frame1=[r for r in data["records"] if r["frame"]==1]
    assert frame1
    assert all(not r["reconstructed"] for r in frame1)
    assert not any(
        e["event_type"]=="internal_miss_recovered"
        and e["start_frame"]<=1<=e["end_frame"]
        for e in data["events"]
    )


def test_default_fast_paths_use_single_model_layout():
    from src.video_text.pipeline import _default_fast_paths

    root=Path("/tmp/subtitle-project")
    repo,checkpoint=_default_fast_paths(root)

    assert repo == root/"model/FAST"
    assert checkpoint == repo/"checkpoints/fast_base_ic15_736_finetune_ic17mlt.pth"
