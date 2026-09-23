from pathlib import Path

import numpy as np

from src.video_text.openvino_backend import (
    OpenVINOTextDetectionPredictor,
    resolve_cpu_engine,
)


class FakeCompiled:
    def __init__(self):
        self.calls = []

    def __call__(self, inputs):
        x = next(iter(inputs.values()))
        self.calls.append(tuple(x.shape))
        return {"prob": np.ones((x.shape[0], 1, 8, 8), dtype=np.float32)}


def fake_preprocess(images):
    batch = np.stack([
        np.full((3, 8, 8), fill_value=i + 1, dtype=np.float32)
        for i, _ in enumerate(images)
    ])
    shapes = [
        np.array([img.shape[0], img.shape[1], 1.0, 1.0], dtype=np.float32)
        for img in images
    ]
    return batch, shapes


def fake_postprocess(pred, shapes, thresh, box_thresh):
    assert pred.shape[0] == len(shapes)
    polys = []
    scores = []
    for i in range(pred.shape[0]):
        polys.append(np.array([[[1, 2], [5, 2], [5, 6], [1, 6]]], dtype=np.int16))
        scores.append([0.9 - i * 0.1])
    return polys, scores


def test_openvino_predictor_preserves_paddleocr_result_contract():
    compiled = FakeCompiled()
    predictor = OpenVINOTextDetectionPredictor(
        model_path=Path("model.xml"),
        compiled_model=compiled,
        input_name="x",
        output_name="prob",
        preprocess_fn=fake_preprocess,
        postprocess_fn=fake_postprocess,
        thresh=.30,
        box_thresh=.50,
    )

    rows = list(predictor.predict([
        np.zeros((20, 30, 3), np.uint8),
        np.zeros((20, 30, 3), np.uint8),
    ]))

    assert compiled.calls == [(2, 3, 8, 8)]
    assert len(rows) == 2
    assert rows[0]["dt_polys"].shape == (1, 4, 2)
    assert rows[0]["dt_scores"].tolist() == [0.9]
    assert rows[1]["dt_scores"].tolist() == [0.8]


def test_auto_cpu_engine_prefers_openvino_ir_then_onnxruntime():
    assert resolve_cpu_engine(
        "auto",
        openvino_available=True,
        openvino_model_exists=True,
        onnxruntime_available=True,
    ) == "openvino"
    assert resolve_cpu_engine(
        "auto",
        openvino_available=True,
        openvino_model_exists=False,
        onnxruntime_available=True,
    ) == "onnxruntime"


def test_explicit_openvino_requires_runtime_and_model():
    try:
        resolve_cpu_engine(
            "openvino",
            openvino_available=False,
            openvino_model_exists=True,
            onnxruntime_available=True,
        )
    except RuntimeError as exc:
        assert "OpenVINO runtime" in str(exc)
    else:
        raise AssertionError("expected missing OpenVINO runtime to fail")



class FakeTensor:
    def __init__(self, value):
        self.data = np.full((1, 1, 8, 8), value, dtype=np.float32)


class FakeRequest:
    def __init__(self, value):
        self.value = value

    def get_output_tensor(self, index):
        assert index == 0
        return FakeTensor(self.value)


class FakeAsyncQueue:
    def __init__(self, compiled_model):
        self.compiled_model = compiled_model
        self.callback = None
        self.pending = []

    def set_callback(self, callback):
        self.callback = callback

    def start_async(self, inputs, userdata=None):
        x = next(iter(inputs.values()))
        self.pending.append((float(x.mean()), userdata))

    def wait_all(self):
        # Deliberately complete in reverse order to verify frame ordering.
        for value, userdata in reversed(self.pending):
            self.callback(FakeRequest(value), userdata)
        self.pending.clear()


def score_from_pred_postprocess(pred, shapes, thresh, box_thresh):
    score = float(np.asarray(pred).mean()) / 10.0
    return (
        [np.array([[[1, 2], [5, 2], [5, 6], [1, 6]]], dtype=np.int16)],
        [[score]],
    )


def test_async_predict_preserves_input_order_when_callbacks_finish_out_of_order():
    compiled = FakeCompiled()
    predictor = OpenVINOTextDetectionPredictor(
        model_path=Path("model.xml"),
        compiled_model=compiled,
        input_name="x",
        output_name="prob",
        preprocess_fn=fake_preprocess,
        postprocess_fn=score_from_pred_postprocess,
        thresh=.30,
        box_thresh=.50,
        async_inference=True,
        async_queue_factory=FakeAsyncQueue,
    )

    rows = list(predictor.predict([
        np.zeros((20, 30, 3), np.uint8),
        np.zeros((20, 30, 3), np.uint8),
        np.zeros((20, 30, 3), np.uint8),
    ]))

    assert [round(float(r["dt_scores"][0]), 3) for r in rows] == [.1, .2, .3]


def test_openvino_cpu_config_leaves_threads_auto_when_zero():
    from src.video_text.openvino_backend import build_openvino_cpu_config

    auto = build_openvino_cpu_config(
        cpu_threads=0,
        performance_hint="THROUGHPUT",
    )
    limited = build_openvino_cpu_config(
        cpu_threads=4,
        performance_hint="THROUGHPUT",
    )

    assert auto == {"PERFORMANCE_HINT": "THROUGHPUT"}
    assert limited == {
        "PERFORMANCE_HINT": "THROUGHPUT",
        "INFERENCE_NUM_THREADS": 4,
    }


def test_static_request_shape_uses_single_image_from_preprocessed_batch():
    from src.video_text.openvino_backend import static_request_shape

    batch = np.zeros((32, 3, 576, 704), dtype=np.float32)

    assert static_request_shape(batch) == [1, 3, 576, 704]
