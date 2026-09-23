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



class CountingResize:
    def __init__(self):
        self.calls = 0

    def __call__(self, imgs):
        self.calls += 1
        return list(imgs), [
            np.array([img.shape[0], img.shape[1], 1.0, 1.0], dtype=np.float32)
            for img in imgs
        ]


class CountingNormalize:
    def __init__(self):
        self.calls = 0

    def __call__(self, imgs):
        self.calls += 1
        return [img.astype(np.float32) for img in imgs]


def test_default_preprocessor_reuses_processor_instances_across_calls():
    from src.video_text.openvino_backend import DefaultTextDetectionPreprocessor

    resize = CountingResize()
    normalize = CountingNormalize()
    pre = DefaultTextDetectionPreprocessor(
        resize_op=resize,
        normalize_op=normalize,
    )
    image = np.zeros((8, 8, 3), dtype=np.uint8)

    pre([image])
    pre([image])

    assert resize.calls == 2
    assert normalize.calls == 2
    assert pre.resize_op is resize
    assert pre.normalize_op is normalize


def test_async_postprocess_starts_before_wait_all_returns():
    import threading

    postprocess_started = threading.Event()

    class OverlapQueue(FakeAsyncQueue):
        def wait_all(self):
            for value, userdata in self.pending:
                self.callback(FakeRequest(value), userdata)
            assert postprocess_started.wait(timeout=1.0)
            self.pending.clear()

    def postprocess(pred, shapes, thresh, box_thresh):
        postprocess_started.set()
        return (
            [np.array([[[1, 2], [5, 2], [5, 6], [1, 6]]], dtype=np.int16)],
            [[0.9]],
        )

    predictor = OpenVINOTextDetectionPredictor(
        model_path=Path("model.xml"),
        compiled_model=FakeCompiled(),
        input_name="x",
        output_name="prob",
        preprocess_fn=fake_preprocess,
        postprocess_fn=postprocess,
        async_inference=True,
        async_queue_factory=OverlapQueue,
    )

    rows = list(predictor.predict([
        np.zeros((20, 30, 3), np.uint8),
        np.zeros((20, 30, 3), np.uint8),
    ]))

    assert len(rows) == 2



def test_openvino_config_accepts_explicit_stream_count_without_limiting_threads():
    from src.video_text.openvino_backend import build_openvino_cpu_config

    config = build_openvino_cpu_config(
        cpu_threads=0,
        performance_hint="THROUGHPUT",
        num_streams=8,
    )

    assert config == {
        "PERFORMANCE_HINT": "THROUGHPUT",
        "NUM_STREAMS": 8,
    }


def test_fused_preprocess_spec_matches_ppocr_mobile_fixed_roi():
    from src.video_text.openvino_backend import fused_preprocess_spec

    spec = fused_preprocess_spec(
        raw_height=576,
        raw_width=720,
        resized_height=576,
        resized_width=704,
    )

    assert spec["tensor_shape"] == [1, 576, 704, 3]
    assert spec["model_shape"] == [1, 3, 576, 704]
    assert spec["tensor_layout"] == "NHWC"
    assert spec["model_layout"] == "NCHW"
    assert np.allclose(spec["mean"], [123.675, 116.28, 103.53])
    assert np.allclose(spec["scale"], [58.395, 57.12, 57.375])


def test_fused_input_keeps_exact_paddle_resize_but_skips_python_normalize():
    from paddlex.inference.models.text_detection.processors import DetResizeForTest
    from src.video_text.openvino_backend import prepare_fused_inputs

    rng = np.random.default_rng(7)
    images = [
        rng.integers(0, 256, size=(576, 720, 3), dtype=np.uint8),
        rng.integers(0, 256, size=(576, 720, 3), dtype=np.uint8),
    ]

    batch, shapes, resized_shape = prepare_fused_inputs(images)
    expected, expected_shapes = DetResizeForTest(
        limit_side_len=960,
        limit_type="max",
    )(imgs=images)

    assert batch.shape == (2, 576, 704, 3)
    assert batch.dtype == np.uint8
    assert resized_shape == (576, 704)
    assert np.array_equal(batch[0], expected[0])
    assert np.array_equal(batch[1], expected[1])
    assert np.allclose(shapes[0], expected_shapes[0])



def test_fused_openvino_model_accepts_raw_uint8_nhwc():
    import pytest

    ov = pytest.importorskip("openvino")
    ops = ov.opset13

    from src.video_text.openvino_backend import build_fused_openvino_model

    param = ops.parameter([1, 3, 4, 4], ov.Type.f32, name="x")
    model = ov.Model([param], [param], "identity")

    fused = build_fused_openvino_model(
        model,
        raw_height=8,
        raw_width=8,
        resized_height=4,
        resized_width=4,
        ov_module=ov,
    )

    assert list(fused.input(0).shape) == [1, 4, 4, 3]
    assert fused.input(0).element_type == ov.Type.u8

    compiled = ov.Core().compile_model(fused, "CPU")
    raw = np.zeros((1, 4, 4, 3), dtype=np.uint8)
    result = compiled({compiled.input(0): raw})[compiled.output(0)]

    assert result.shape == (1, 3, 4, 4)



def test_async_fused_predict_submits_raw_uint8_nhwc():
    class RecordingQueue(FakeAsyncQueue):
        last = None

        def __init__(self, compiled_model):
            super().__init__(compiled_model)
            self.inputs = []
            RecordingQueue.last = self

        def start_async(self, inputs, userdata=None):
            x = next(iter(inputs.values()))
            self.inputs.append((tuple(x.shape), x.dtype))
            super().start_async(inputs, userdata=userdata)

    predictor = OpenVINOTextDetectionPredictor(
        model_path=Path("model.xml"),
        compiled_model=FakeCompiled(),
        input_name="x",
        output_name="prob",
        preprocess_fn=fake_preprocess,
        postprocess_fn=score_from_pred_postprocess,
        async_inference=True,
        fuse_preprocess=True,
        async_queue_factory=RecordingQueue,
    )

    rows = list(predictor.predict([
        np.zeros((64, 96, 3), np.uint8),
        np.ones((64, 96, 3), np.uint8),
    ]))

    assert len(rows) == 2
    assert RecordingQueue.last.inputs == [
        ((1, 64, 96, 3), np.dtype("uint8")),
        ((1, 64, 96, 3), np.dtype("uint8")),
    ]
