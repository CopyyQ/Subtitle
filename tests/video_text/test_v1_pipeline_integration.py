from pathlib import Path
import json

import cv2
import numpy as np

from src.video_text.pipeline import PipelineConfig, SubtitlePipeline
from src.video_text.types import Candidate


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "tests/video_text/.tmp_v1_pipeline"


class JitterBackend:
    batch_size = 1

    def __init__(self):
        self.i = 0

    def detect_batch(self, images):
        out = []
        for _ in images:
            i = self.i
            self.i += 1
            dx = (i % 5) - 2
            dy = (i % 3) - 1
            out.append(([
                Candidate(
                    np.array([58 + dx, 10 + dy, 270, 58], np.float32),
                    .96, "HIGH", "fake",
                ),
                Candidate(
                    np.array([138, 63, 182 + (i % 2), 98], np.float32),
                    .94, "HIGH", "fake",
                ),
            ], []))
        return out


def _outlined_rect(frame, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), (10, 10, 10), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 245, 245), -1)


def _make_video(path, n=12):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 220)
    )
    assert writer.isOpened()
    for _ in range(n):
        frame = np.full((220, 320, 3), 100, np.uint8)
        for x in (65, 105, 145, 185, 225):
            _outlined_rect(frame, x, 142, x + 22, 175)
        _outlined_rect(frame, 145, 183, 175, 214)
        writer.write(frame)
    writer.release()


def test_pipeline_v1_locks_static_line_geometry_across_lifecycle():
    src = WORK / "static_jitter.mp4"
    _make_video(src)
    out = WORK / "v1.mp4"
    cfg = PipelineConfig(detector="fast",
        temporal_mode="v1",
        validate_chinese=False,
        roi_bottom_fraction=.45,
        output_codec="h264",
    )
    result = SubtitlePipeline(cfg, backend=JitterBackend()).run(
        src, out, max_frames=12
    )
    data = json.loads(result.coordinate_json.read_text())
    assert data["metadata"]["temporal_mode"] == "v1"
    assert data["metadata"]["v1_static_locked_track_count"] >= 2
    assert data["metadata"]["v1_postlock_max_edge_range_px"] == 0.0
    assert data["metadata"]["v1_final_overlap_frame_count"] == 0

    by_track = {}
    for row in data["records"]:
        by_track.setdefault(row["track_id"], []).append(tuple(row["bbox"]))
    assert by_track
    assert all(len(set(boxes)) == 1 for boxes in by_track.values())


def test_v1_detection_only_never_constructs_recognizer(monkeypatch):
    import src.video_text.pipeline as pipeline_module

    src = WORK / "no_recognizer.mp4"
    _make_video(src, n=4)
    out = WORK / "no_recognizer_out.mp4"

    class ForbiddenRecognizer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("recognizer must not be constructed")

    monkeypatch.setattr(
        pipeline_module, "EasyOCRChineseRecognizer", ForbiddenRecognizer
    )
    cfg = PipelineConfig(detector="fast",
        temporal_mode="v1",
        validate_chinese=False,
        roi_bottom_fraction=.45,
        output_codec="h264",
    )
    result = SubtitlePipeline(cfg, backend=JitterBackend()).run(
        src, out, max_frames=4
    )
    assert result.output_video.exists()


def test_v1_recognizer_only_validation_skips_detector_based_ocr_tightening(monkeypatch):
    import src.video_text.pipeline as pipeline_module

    src = WORK / "recognizer_only.mp4"
    _make_video(src, n=4)
    out = WORK / "recognizer_only_out.mp4"

    class RecognizerOnly:
        supports_detection = False
        def recognize(self, crop):
            return "我们", .95

    def forbidden_ocr_tighten(*args, **kwargs):
        raise AssertionError("detector-based OCR tightening must be skipped")

    def glyph_tighten_without_ocr(video_path, tracks, identity, recognizer, **kwargs):
        assert recognizer is None
        return {
            "glyph_tightened_track_count":0,
            "glyph_tightening_area_ratio_median":1.0,
            "glyph_center_alignment_adjustment_count":0,
        }

    monkeypatch.setattr(
        pipeline_module,
        "tighten_v1_static_tracks_with_ocr",
        forbidden_ocr_tighten,
    )
    monkeypatch.setattr(
        pipeline_module,
        "tighten_v1_static_tracks_with_temporal_glyphs",
        glyph_tighten_without_ocr,
    )
    cfg = PipelineConfig(detector="fast",
        temporal_mode="v1",
        validate_chinese=True,
        roi_bottom_fraction=.45,
        output_codec="h264",
    )
    result = SubtitlePipeline(
        cfg,
        backend=JitterBackend(),
        recognizer=RecognizerOnly(),
    ).run(src, out, max_frames=4)
    assert result.output_video.exists()
    assert result.metrics["v1_ocr_tightened_track_count"] == 0


def test_v1_signature_differs_from_v55_for_same_source():
    src = WORK / "signature.mp4"
    _make_video(src, n=1)
    p55 = SubtitlePipeline(
        PipelineConfig(detector="fast", temporal_mode="v5_5", validate_chinese=False),
        backend=JitterBackend(),
    )
    p56 = SubtitlePipeline(
        PipelineConfig(detector="fast", temporal_mode="v1", validate_chinese=False),
        backend=JitterBackend(),
    )
    s55 = p55._signature(src, 1)
    s56 = p56._signature(src, 1)
    assert s55 != s56
    assert s56["temporal_mode"] == "v1"
