import numpy as np
import pytest

from src.video_text.ppocrv5_backend import PPOCRv5MobileBackend


class FakePredictor:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def predict(self, images):
        self.calls.append(len(images))
        return self.rows[:len(images)]


def result(polys, scores):
    return {
        "dt_polys": [np.asarray(p, dtype=np.float32) for p in polys],
        "dt_scores": np.asarray(scores, dtype=np.float32),
    }


def test_maps_polygons_to_high_and_low_candidates():
    predictor = FakePredictor([
        result(
            [
                [[10, 10], [110, 10], [110, 40], [10, 40]],
                [[20, 50], [50, 50], [50, 80], [20, 80]],
            ],
            [.93, .61],
        )
    ])
    backend = PPOCRv5MobileBackend(
        device="cpu",
        batch_size=4,
        high_score=.84,
        low_score=.50,
        predictor=predictor,
    )

    high, low = backend.detect_batch(
        [np.zeros((100, 200, 3), np.uint8)]
    )[0]

    assert high[0].bbox.tolist() == [10, 10, 110, 40]
    assert low[0].bbox.tolist() == [20, 50, 50, 80]
    assert high[0].source == "ppocrv5_mobile"
    assert low[0].source == "ppocrv5_mobile"


def test_drops_scores_below_low_threshold_and_preserves_batch_length():
    predictor = FakePredictor([
        result([[[1, 2], [8, 2], [8, 9], [1, 9]]], [.49]),
        result([[[2, 3], [9, 3], [9, 10], [2, 10]]], [.99]),
    ])
    backend = PPOCRv5MobileBackend(
        device="cpu",
        batch_size=2,
        high_score=.84,
        low_score=.50,
        predictor=predictor,
    )

    rows = backend.detect_batch([
        np.zeros((20, 20, 3), np.uint8),
        np.zeros((20, 20, 3), np.uint8),
    ])

    assert len(rows) == 2
    assert rows[0] == ([], [])
    assert len(rows[1][0]) == 1
    assert rows[1][1] == []


def test_rejects_predictor_batch_length_mismatch():
    backend = PPOCRv5MobileBackend(
        device="cpu",
        predictor=FakePredictor([]),
    )
    with pytest.raises(RuntimeError, match="batch length"):
        backend.detect_batch([np.zeros((20, 20, 3), np.uint8)])


def test_threshold_validation():
    with pytest.raises(ValueError, match="low_score"):
        PPOCRv5MobileBackend(
            device="cpu", high_score=.4, low_score=.5,
            predictor=FakePredictor([]),
        )


def test_cpu_backend_disables_mkldnn_for_paddle_3_3_pir_compat(monkeypatch):
    import sys
    import types
    import src.video_text.ppocrv5_backend as backend_module

    captured = {}

    class FakeTextDetection:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        types.SimpleNamespace(TextDetection=FakeTextDetection),
    )
    monkeypatch.setattr(
        backend_module,
        "_onnxruntime_available",
        lambda: False,
    )

    PPOCRv5MobileBackend(device="cpu", predictor=None)

    assert captured["device"] == "cpu"
    assert captured["enable_mkldnn"] is False


def test_cpu_backend_forwards_explicit_cpu_threads(monkeypatch):
    import sys
    import types

    captured = {}

    class FakeTextDetection:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        types.SimpleNamespace(TextDetection=FakeTextDetection),
    )

    PPOCRv5MobileBackend(device="cpu", cpu_threads=12, predictor=None)

    assert captured["cpu_threads"] == 12


def test_cpu_backend_prefers_onnxruntime_when_available(monkeypatch):
    import sys
    import types
    import src.video_text.ppocrv5_backend as backend_module

    captured = {}

    class FakeTextDetection:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "paddleocr",
        types.SimpleNamespace(TextDetection=FakeTextDetection),
    )
    monkeypatch.setattr(
        backend_module,
        "_onnxruntime_available",
        lambda: True,
    )

    PPOCRv5MobileBackend(device="cpu", predictor=None)

    assert captured["engine"] == "onnxruntime"
    assert "enable_mkldnn" not in captured


def test_cpu_backend_uses_explicit_openvino_ir(monkeypatch, tmp_path):
    import src.video_text.ppocrv5_backend as backend_module

    model = tmp_path / "model.xml"
    model.write_text("<xml/>")
    captured = {}

    class FakeOpenVINOPredictor:
        def __init__(self, model_path, **kwargs):
            captured["model_path"] = str(model_path)
            captured.update(kwargs)

        def predict(self, images):
            return []

    monkeypatch.setattr(
        backend_module,
        "OpenVINOTextDetectionPredictor",
        FakeOpenVINOPredictor,
        raising=False,
    )
    monkeypatch.setattr(
        backend_module,
        "_openvino_available",
        lambda: True,
        raising=False,
    )

    backend = PPOCRv5MobileBackend(
        device="cpu",
        cpu_engine="openvino",
        openvino_model=model,
        cpu_threads=6,
        predictor=None,
    )

    assert backend.cpu_engine == "openvino"
    assert captured["model_path"] == str(model)
    assert captured["cpu_threads"] == 6
    assert captured["async_inference"] is True
    assert captured["thresh"] == .30
    assert captured["box_thresh"] == .50



def test_openvino_backend_forwards_streams_and_fused_preprocess(monkeypatch, tmp_path):
    import src.video_text.ppocrv5_backend as backend_module

    model = tmp_path / "model.xml"
    model.write_text("<xml/>")
    captured = {}

    class FakeOpenVINOPredictor:
        def __init__(self, model_path, **kwargs):
            captured.update(kwargs)

        def predict(self, images):
            return []

    monkeypatch.setattr(
        backend_module,
        "OpenVINOTextDetectionPredictor",
        FakeOpenVINOPredictor,
    )
    monkeypatch.setattr(backend_module, "_openvino_available", lambda: True)

    PPOCRv5MobileBackend(
        device="cpu",
        cpu_engine="openvino",
        openvino_model=model,
        openvino_num_streams=8,
        openvino_fuse_preprocess=True,
        predictor=None,
    )

    assert captured["num_streams"] == 8
    assert captured["fuse_preprocess"] is True


def test_adaptive_gate_reuses_stable_results_and_reduces_inference_count():
    row = result([[[1, 2], [8, 2], [8, 9], [1, 9]]], [.95])
    predictor = FakePredictor([row, row])
    backend = PPOCRv5MobileBackend(
        device="cpu",
        predictor=predictor,
        adaptive_gating=True,
        gate_max_skip_frames=2,
        gate_change_threshold=.02,
    )
    image = np.full((120, 200, 3), 20, dtype=np.uint8)

    rows = backend.detect_batch([image, image.copy(), image.copy(), image.copy()])

    assert len(rows) == 4
    assert predictor.calls == [2]
    assert all(len(high) == 1 for high, _ in rows)
    assert backend.gate_total_frames == 4
    assert backend.gate_inferred_frames == 2
    assert backend.gate_skipped_frames == 2


def test_adaptive_gate_detects_visual_change_immediately():
    row_a = result([[[1, 2], [8, 2], [8, 9], [1, 9]]], [.95])
    row_b = result([[[11, 12], [18, 12], [18, 19], [11, 19]]], [.96])
    predictor = FakePredictor([row_a, row_b])
    backend = PPOCRv5MobileBackend(
        device="cpu",
        predictor=predictor,
        adaptive_gating=True,
        gate_max_skip_frames=5,
        gate_change_threshold=.02,
    )
    base = np.full((120, 200, 3), 20, dtype=np.uint8)
    changed = base.copy()
    changed[40:80, 60:140] = 240

    rows = backend.detect_batch([base, changed, changed.copy()])

    assert predictor.calls == [2]
    assert rows[0][0][0].bbox.tolist() == [1, 2, 8, 9]
    assert rows[1][0][0].bbox.tolist() == [11, 12, 18, 19]
    assert rows[2][0][0].bbox.tolist() == [11, 12, 18, 19]
