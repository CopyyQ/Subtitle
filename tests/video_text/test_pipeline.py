from pathlib import Path
import json
import cv2
import numpy as np
import torch
from src.video_text.types import Candidate
from src.video_text.io_probe import probe_video
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
    cfg=PipelineConfig(detector="fast",validate_chinese=False,roi_bottom_fraction=.45,
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
    for key in (
        "detection_loop_seconds","temporal_postprocess_seconds",
        "render_encode_seconds","source_duration_seconds",
        "real_time_factor","detector_latency_mean_ms",
        "detector_latency_p50_ms","detector_latency_p95_ms",
        "output_frame_count","dropped_frame_count",
        "device","precision","batch_size","cpu_threads",
    ):
        assert key in r1.metrics


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
        detector="fast",
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
        detector="fast",
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


def test_cpu_detection_never_synchronizes_cuda_when_cuda_is_present(monkeypatch):
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"cpu_no_cuda_sync.mp4"
    make_video(src,n=2)
    backend=FakeBackend({})
    backend.device=torch.device("cpu")
    backend.batch_size=2
    sync_calls=[]
    monkeypatch.setattr(torch.cuda,"is_available",lambda: True)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda *a,**k: sync_calls.append(True))
    pipeline=SubtitlePipeline(
        PipelineConfig(
            validate_chinese=False,
            device="cpu",
            batch_size=2,
        ),
        backend=backend,
    )
    info=probe_video(src)
    cache=WORK/"cpu_no_cuda_sync.cache.json"
    cache.unlink(missing_ok=True)
    frames,_,_=pipeline._detect(src,info,2,cache)
    assert len(frames)==2
    assert sync_calls==[]


def test_cpu_full_run_does_not_touch_cuda_runtime_when_cuda_is_present(monkeypatch):
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"cpu_no_cuda_runtime.mp4"
    make_video(src,n=2)
    backend=FakeBackend({})
    backend.device=torch.device("cpu")
    backend.batch_size=2
    calls=[]
    monkeypatch.setattr(torch.cuda,"is_available",lambda: True)
    monkeypatch.setattr(torch.cuda,"synchronize",lambda *a,**k: calls.append("sync"))
    monkeypatch.setattr(torch.cuda,"empty_cache",lambda *a,**k: calls.append("empty"))
    monkeypatch.setattr(torch.cuda,"reset_peak_memory_stats",lambda *a,**k: calls.append("reset"))
    monkeypatch.setattr(torch.cuda,"max_memory_allocated",lambda *a,**k: calls.append("max") or 0)
    out=WORK/"cpu_no_cuda_runtime_out.mp4"
    cache=out.parent/f"{src.stem}.video_text_cache.json"
    cache.unlink(missing_ok=True)
    result=SubtitlePipeline(
        PipelineConfig(validate_chinese=False,device="cpu",batch_size=2),
        backend=backend,
    ).run(src,out,max_frames=2)
    assert result.metrics["peak_vram_mb"]==0.0
    assert calls==[]


def test_ppocr_pipeline_uses_light_temporal_path_and_labels_low_backfill():
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"ppocr_integration.mp4"
    make_video(src,n=4)
    box=[20,4,95,20]
    timeline={
        0:([],[c(box,.65,"LOW")]),
        1:([c(box,.94,"HIGH")],[]),
        2:([c(box,.95,"HIGH")],[]),
        3:([c(box,.95,"HIGH")],[]),
    }
    backend=FakeBackend(timeline)
    backend.device=torch.device("cpu")
    backend.batch_size=4
    out=WORK/"ppocr_integration_out.mp4"
    cache=out.parent/f"{src.stem}.video_text_cache.json"
    cache.unlink(missing_ok=True)

    result=SubtitlePipeline(
        PipelineConfig(
            detector="ppocrv5_mobile",
            device="cpu",
            validate_chinese=False,
            roi_bottom_fraction=.45,
        ),
        backend=backend,
    ).run(src,out,max_frames=4)

    data=json.loads(result.coordinate_json.read_text())
    assert result.metrics["detector_name"]=="PP-OCRv5_mobile_det"
    assert result.metrics["ppocr_track_count"]==1
    assert len(data["frames"])==4
    frame0=data["frames"][0]["boxes"]
    assert len(frame0)==1
    assert frame0[0]["source"]=="PP_LOW_BACKFILL"
    assert data["metadata"]["ppocr_thresh"]==.30
    assert data["metadata"]["ppocr_box_thresh"]==.50


def test_cache_signature_changes_with_detector_and_ppocr_thresholds(tmp_path):
    source=tmp_path/"source.mp4"
    source.write_bytes(b"cache-signature")
    pp_a=SubtitlePipeline(PipelineConfig(
        detector="ppocrv5_mobile",
        ppocr_thresh=.30,
        ppocr_box_thresh=.50,
    ))._signature(source,10)
    pp_b=SubtitlePipeline(PipelineConfig(
        detector="ppocrv5_mobile",
        ppocr_thresh=.20,
        ppocr_box_thresh=.40,
    ))._signature(source,10)
    fast=SubtitlePipeline(PipelineConfig(
        detector="fast",
        high_score=.84,
        low_score=.50,
    ))._signature(source,10)

    assert pp_a!=pp_b
    assert pp_a!=fast
    assert pp_a["detector_model"]=="PP-OCRv5_mobile_det"


def test_ppocr_gpu_detection_prefetches_without_torch_cuda_sync(monkeypatch):
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"ppocr_prefetch_no_sync.mp4"
    make_video(src,n=8)
    backend=FakeBackend({})
    backend.device=torch.device("cuda")
    backend.batch_size=4
    sync_calls=[]
    monkeypatch.setattr(
        "src.video_text.pipeline.synchronize",
        lambda device: sync_calls.append(str(device)),
    )
    pipeline=SubtitlePipeline(
        PipelineConfig(
            detector="ppocrv5_mobile",
            device="cuda",
            batch_size=4,
            decode_prefetch_batches=2,
            validate_chinese=False,
        ),
        backend=backend,
    )
    info=probe_video(src)
    cache=WORK/"ppocr_prefetch_no_sync.cache.json"
    cache.unlink(missing_ok=True)
    frames,_,_=pipeline._detect(src,info,8,cache)
    assert [f.frame_index for f in frames]==list(range(8))
    assert backend.calls==2
    assert sync_calls==[]
