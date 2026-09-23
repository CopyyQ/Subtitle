from __future__ import annotations

import importlib.util
from pathlib import Path
import queue
import threading
from typing import Callable, Iterable, Sequence

import numpy as np


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def resolve_cpu_engine(
    requested: str,
    *,
    openvino_available: bool | None = None,
    openvino_model_exists: bool = False,
    onnxruntime_available: bool | None = None,
) -> str:
    requested = str(requested).lower()
    if requested not in {"auto", "openvino", "onnxruntime", "paddle"}:
        raise ValueError("cpu engine must be auto, openvino, onnxruntime, or paddle")

    ov_ok = _module_available("openvino") if openvino_available is None else bool(openvino_available)
    ort_ok = (
        _module_available("onnxruntime")
        if onnxruntime_available is None
        else bool(onnxruntime_available)
    )

    if requested == "openvino":
        if not ov_ok:
            raise RuntimeError("OpenVINO runtime is not installed")
        if not openvino_model_exists:
            raise RuntimeError("OpenVINO IR model was not found")
        return "openvino"
    if requested == "onnxruntime":
        if not ort_ok:
            raise RuntimeError("ONNX Runtime is not installed")
        return "onnxruntime"
    if requested == "paddle":
        return "paddle"

    if ov_ok and openvino_model_exists:
        return "openvino"
    if ort_ok:
        return "onnxruntime"
    return "paddle"


def build_openvino_cpu_config(
    *,
    cpu_threads: int = 0,
    performance_hint: str = "THROUGHPUT",
    num_streams: int = 0,
) -> dict:
    config = {"PERFORMANCE_HINT": str(performance_hint).upper()}
    if int(cpu_threads) > 0:
        config["INFERENCE_NUM_THREADS"] = int(cpu_threads)
    if int(num_streams) > 0:
        config["NUM_STREAMS"] = int(num_streams)
    return config


def _resize_shape_for_ppocr(raw_height: int, raw_width: int) -> tuple[int, int]:
    h = int(raw_height)
    w = int(raw_width)
    ratio = 1.0
    if max(h, w) > 960:
        ratio = 960.0 / max(h, w)
    resize_h = max(int(round((h * ratio) / 32.0) * 32), 32)
    resize_w = max(int(round((w * ratio) / 32.0) * 32), 32)
    return resize_h, resize_w


def fused_preprocess_spec(
    *,
    raw_height: int,
    raw_width: int,
    resized_height: int,
    resized_width: int,
) -> dict:
    return {
        # Resize stays in OpenCV/Paddle for exact DB-detector parity. The
        # OpenVINO graph receives resized uint8 NHWC and fuses the remaining
        # convert/normalize/layout preprocessing.
        "tensor_shape": [1, int(resized_height), int(resized_width), 3],
        "model_shape": [1, 3, int(resized_height), int(resized_width)],
        "tensor_layout": "NHWC",
        "model_layout": "NCHW",
        "mean": [123.675, 116.28, 103.53],
        "scale": [58.395, 57.12, 57.375],
    }


def build_fused_openvino_model(
    model,
    *,
    raw_height: int,
    raw_width: int,
    resized_height: int,
    resized_width: int,
    ov_module=None,
):
    if ov_module is None:
        import openvino as ov
        ov_module = ov

    spec = fused_preprocess_spec(
        raw_height=raw_height,
        raw_width=raw_width,
        resized_height=resized_height,
        resized_width=resized_width,
    )
    preprocess = ov_module.preprocess.PrePostProcessor(model)
    inp = preprocess.input()
    inp.tensor().set_element_type(ov_module.Type.u8)
    inp.tensor().set_shape(spec["tensor_shape"])
    inp.tensor().set_layout(ov_module.Layout(spec["tensor_layout"]))
    inp.model().set_layout(ov_module.Layout(spec["model_layout"]))

    steps = inp.preprocess()
    steps.convert_element_type(ov_module.Type.f32)
    steps.mean(spec["mean"])
    steps.scale(spec["scale"])
    return preprocess.build()


def prepare_fused_inputs(images: Sequence[np.ndarray], resize_op=None):
    if not images:
        return np.empty((0, 0, 0, 3), dtype=np.uint8), [], (0, 0)
    h, w = images[0].shape[:2]
    for image in images:
        if image.shape[:2] != (h, w):
            raise ValueError("fused OpenVINO preprocessing requires a fixed ROI shape")
    if resize_op is None:
        from paddlex.inference.models.text_detection.processors import DetResizeForTest

        resize_op = DetResizeForTest(limit_side_len=960, limit_type="max")
    resized, shapes = resize_op(imgs=list(images))
    batch = np.stack(
        [np.asarray(image, dtype=np.uint8) for image in resized],
        axis=0,
    )
    resized_h, resized_w = batch.shape[1:3]
    return batch, shapes, (int(resized_h), int(resized_w))


def static_request_shape(batch: np.ndarray) -> list[int]:
    batch = np.asarray(batch)
    if batch.ndim != 4:
        raise ValueError("OpenVINO detector batch must be NCHW rank 4")
    return [1, *[int(x) for x in batch.shape[1:]]]


class DefaultTextDetectionPreprocessor:
    def __init__(self, *, resize_op=None, normalize_op=None):
        if resize_op is None or normalize_op is None:
            from paddlex.inference.models.text_detection.processors import (
                DetResizeForTest,
                NormalizeImage,
            )
        self.resize_op = resize_op or DetResizeForTest(
            limit_side_len=960,
            limit_type="max",
        )
        self.normalize_op = normalize_op or NormalizeImage(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
            scale=1.0 / 255.0,
            order="hwc",
        )

    def __call__(self, images: Sequence[np.ndarray]):
        resized, shapes = self.resize_op(imgs=list(images))
        normalized = self.normalize_op(imgs=resized)
        chw = [
            np.transpose(img, (2, 0, 1)).astype(np.float32, copy=False)
            for img in normalized
        ]
        return np.stack(chw, axis=0), shapes


def _default_preprocess(images: Sequence[np.ndarray]):
    return DefaultTextDetectionPreprocessor()(images)


class _DefaultPostprocess:
    def __init__(self, thresh: float, box_thresh: float):
        from paddlex.inference.models.text_detection.processors import DBPostProcess

        self.op = DBPostProcess(
            thresh=thresh,
            box_thresh=box_thresh,
            max_candidates=1000,
            unclip_ratio=1.5,
            use_dilation=False,
            score_mode="fast",
            box_type="quad",
        )

    def __call__(self, pred, shapes, thresh, box_thresh):
        return self.op(
            [pred],
            shapes,
            thresh=thresh,
            box_thresh=box_thresh,
            unclip_ratio=1.5,
        )


class OpenVINOTextDetectionPredictor:
    """PP-OCRv5 Mobile detector runner backed by OpenVINO CPU inference.

    Returns the same dt_polys/dt_scores contract as PaddleOCR TextDetection.
    """

    def __init__(
        self,
        model_path: str | Path,
        *,
        thresh: float = .30,
        box_thresh: float = .50,
        cpu_threads: int = 0,
        performance_hint: str = "THROUGHPUT",
        async_inference: bool = False,
        num_streams: int = 0,
        fuse_preprocess: bool = False,
        compiled_model=None,
        input_name=None,
        output_name=None,
        preprocess_fn: Callable | None = None,
        postprocess_fn: Callable | None = None,
        async_queue_factory=None,
    ):
        self.model_path = Path(model_path)
        self.thresh = float(thresh)
        self.box_thresh = float(box_thresh)
        self.cpu_threads = int(cpu_threads)
        self.performance_hint = str(performance_hint).upper()
        self.async_inference = bool(async_inference)
        self.requested_num_streams = int(num_streams)
        self.fuse_preprocess = bool(fuse_preprocess)
        self.preprocess_fn = preprocess_fn or DefaultTextDetectionPreprocessor()
        self._fused_resize_op = None
        if self.fuse_preprocess:
            from paddlex.inference.models.text_detection.processors import DetResizeForTest

            self._fused_resize_op = DetResizeForTest(
                limit_side_len=960,
                limit_type="max",
            )
        self.postprocess_fn = postprocess_fn or _DefaultPostprocess(
            self.thresh, self.box_thresh
        )
        self._async_queue_factory = async_queue_factory
        self._core = None
        self._compiled_shape = None
        self._async_queue = None
        self.num_streams = None
        self.optimal_requests = None
        self.inference_num_threads = None

        if compiled_model is None:
            import openvino as ov

            self._ov = ov
            self._core = ov.Core()
            if self.async_inference:
                # Compile lazily after seeing the first real preprocessed shape.
                self.compiled_model = None
                self.input_name = None
                self.output_name = None
            else:
                model = self._core.read_model(str(self.model_path))
                config = build_openvino_cpu_config(
                    cpu_threads=self.cpu_threads,
                    performance_hint=self.performance_hint,
                )
                compiled_model = self._core.compile_model(model, "CPU", config)
                self.compiled_model = compiled_model
                self.input_name = compiled_model.input(0)
                self.output_name = compiled_model.output(0)
        else:
            self._ov = None
            self.compiled_model = compiled_model
            self.input_name = input_name if input_name is not None else "x"
            self.output_name = output_name if output_name is not None else "prob"
            if self.async_inference:
                self._build_async_queue()

    def _get_property(self, name):
        if self.compiled_model is None:
            return None
        try:
            return self.compiled_model.get_property(name)
        except Exception:
            return None

    def _record_runtime_properties(self):
        self.num_streams = self._get_property("NUM_STREAMS")
        self.optimal_requests = self._get_property(
            "OPTIMAL_NUMBER_OF_INFER_REQUESTS"
        )
        self.inference_num_threads = self._get_property("INFERENCE_NUM_THREADS")

    def _build_async_queue(self):
        if self.compiled_model is None:
            raise RuntimeError("compiled OpenVINO model is required")
        if self._async_queue_factory is not None:
            async_queue = self._async_queue_factory(self.compiled_model)
        else:
            import openvino as ov

            async_queue = ov.AsyncInferQueue(self.compiled_model)

        def callback(request, userdata):
            result_queue, index, shape_meta = userdata
            raw = np.asarray(request.get_output_tensor(0).data).copy()
            result_queue.put((index, raw, shape_meta))

        async_queue.set_callback(callback)
        self._async_queue = async_queue
        self._record_runtime_properties()

    def _compile_async_for_shape(self, request_shape, resized_shape=None):
        request_shape = [int(x) for x in request_shape]
        cache_key = (
            tuple(request_shape),
            tuple(int(x) for x in resized_shape) if resized_shape is not None else None,
        )
        if (
            self.compiled_model is not None
            and self._compiled_shape == cache_key
        ):
            return
        if self._core is None:
            # Injected compiled models are assumed to already accept the shape.
            if self.compiled_model is None:
                raise RuntimeError("OpenVINO core unavailable for compilation")
            self._compiled_shape = cache_key
            if self._async_queue is None:
                self._build_async_queue()
            return

        model = self._core.read_model(str(self.model_path))
        if self.fuse_preprocess:
            if resized_shape is None:
                raise RuntimeError("resized_shape is required for fused preprocessing")
            raw_h, raw_w = int(request_shape[1]), int(request_shape[2])
            resized_h, resized_w = [int(x) for x in resized_shape]
            model.reshape({model.input(0): [1, 3, resized_h, resized_w]})
            model = build_fused_openvino_model(
                model,
                raw_height=raw_h,
                raw_width=raw_w,
                resized_height=resized_h,
                resized_width=resized_w,
                ov_module=self._ov,
            )
        else:
            model.reshape({model.input(0): request_shape})
        config = build_openvino_cpu_config(
            cpu_threads=self.cpu_threads,
            performance_hint=self.performance_hint,
            num_streams=self.requested_num_streams,
        )
        self.compiled_model = self._core.compile_model(model, "CPU", config)
        self.input_name = self.compiled_model.input(0)
        self.output_name = self.compiled_model.output(0)
        self._compiled_shape = cache_key
        self._build_async_queue()

    def _extract_output(self, result):
        if self.output_name in result:
            return np.asarray(result[self.output_name])
        if isinstance(self.output_name, str):
            for key, value in result.items():
                names = set()
                try:
                    names = set(key.get_names())
                except Exception:
                    pass
                if self.output_name in names:
                    return np.asarray(value)
        if len(result) == 1:
            return np.asarray(next(iter(result.values())))
        raise RuntimeError("unable to resolve OpenVINO detector output")

    def _rows_from_single_outputs(self, outputs):
        rows = [None] * len(outputs)
        for index, pred, shape_meta in outputs:
            polys, scores = self.postprocess_fn(
                pred,
                [shape_meta],
                self.thresh,
                self.box_thresh,
            )
            rows[index] = {
                "dt_polys": np.asarray(polys[0]),
                "dt_scores": np.asarray(scores[0]),
            }
        return rows

    def _predict_async(self, batch, shapes, resized_shape=None):
        if self.fuse_preprocess:
            request_shape = [1, *[int(x) for x in batch.shape[1:]]]
        else:
            request_shape = static_request_shape(batch)
        self._compile_async_for_shape(request_shape, resized_shape=resized_shape)

        raw_queue = queue.Queue()
        sentinel = object()
        rows = [None] * len(shapes)
        errors = []

        def postprocess_worker():
            while True:
                item = raw_queue.get()
                if item is sentinel:
                    return
                index, pred, shape_meta = item
                try:
                    polys, scores = self.postprocess_fn(
                        pred,
                        [shape_meta],
                        self.thresh,
                        self.box_thresh,
                    )
                    rows[index] = {
                        "dt_polys": np.asarray(polys[0]),
                        "dt_scores": np.asarray(scores[0]),
                    }
                except BaseException as exc:
                    errors.append(exc)

        worker = threading.Thread(
            target=postprocess_worker,
            name="openvino-db-postprocess",
            daemon=True,
        )
        worker.start()
        try:
            for index, shape_meta in enumerate(shapes):
                sample = batch[index:index + 1]
                self._async_queue.start_async(
                    {self.input_name: sample},
                    userdata=(raw_queue, index, shape_meta),
                )
            self._async_queue.wait_all()
        finally:
            raw_queue.put(sentinel)
            worker.join()

        if errors:
            raise errors[0]
        if any(row is None for row in rows):
            raise RuntimeError("OpenVINO async postprocess returned incomplete rows")
        return rows

    def predict(self, images: Sequence[np.ndarray]) -> Iterable[dict]:
        if not images:
            return iter(())
        resized_shape = None
        if self.async_inference and self.fuse_preprocess:
            batch, shapes, resized_shape = prepare_fused_inputs(
                images,
                resize_op=self._fused_resize_op,
            )
        else:
            batch, shapes = self.preprocess_fn(images)

        if self.async_inference:
            return iter(
                self._predict_async(
                    batch,
                    shapes,
                    resized_shape=resized_shape,
                )
            )

        result = self.compiled_model({self.input_name: batch})
        pred = self._extract_output(result)
        polys, scores = self.postprocess_fn(
            pred, shapes, self.thresh, self.box_thresh
        )
        rows = []
        for p, s in zip(polys, scores):
            rows.append(
                {
                    "dt_polys": np.asarray(p),
                    "dt_scores": np.asarray(s),
                }
            )
        return iter(rows)
