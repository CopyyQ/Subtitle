from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .openvino_backend import _DefaultPostprocess, prepare_fused_inputs


def default_ppocrv5_onnx_model() -> Path | None:
    bundled = Path(__file__).resolve().parents[2] / "models" / "PP-OCRv5_mobile_det" / "inference.onnx"
    if bundled.is_file():
        return bundled
    cached = Path.home() / ".paddlex" / "official_models" / "PP-OCRv5_mobile_det" / "inference.onnx"
    return cached if cached.is_file() else None


def _fused_model_path(source: Path) -> Path:
    return source.with_name(f"{source.stem}.uint8_nhwc{source.suffix}")


def ensure_fused_uint8_nhwc_model(source: str | Path) -> Path:
    source = Path(source)
    target = _fused_model_path(source)
    if target.is_file() and target.stat().st_mtime_ns >= source.stat().st_mtime_ns:
        return target

    import onnx
    from onnx import TensorProto, helper, numpy_helper

    model = onnx.load(str(source))
    graph = model.graph
    input_meta = graph.input[0]
    if input_meta.type.tensor_type.elem_type == TensorProto.UINT8:
        return source

    old_name = input_meta.name
    new_name = "images_uint8_nhwc"
    del graph.input[:]
    graph.input.append(
        helper.make_tensor_value_info(new_name, TensorProto.UINT8, ["N", "H", "W", 3])
    )

    alpha = np.asarray(
        [1.0 / 255.0 / 0.229, 1.0 / 255.0 / 0.224, 1.0 / 255.0 / 0.225],
        dtype=np.float32,
    ).reshape(1, 1, 1, 3)
    beta = np.asarray(
        [-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
        dtype=np.float32,
    ).reshape(1, 1, 1, 3)
    graph.initializer.extend(
        [numpy_helper.from_array(alpha, "pre_alpha"), numpy_helper.from_array(beta, "pre_beta")]
    )
    nodes = [
        helper.make_node("Cast", [new_name], ["pre_float"], to=TensorProto.FLOAT, name="PreCast"),
        helper.make_node("Mul", ["pre_float", "pre_alpha"], ["pre_mul"], name="PreMul"),
        helper.make_node("Add", ["pre_mul", "pre_beta"], ["pre_hwc"], name="PreAdd"),
        helper.make_node("Transpose", ["pre_hwc"], [old_name], perm=[0, 3, 1, 2], name="PreTranspose"),
    ]
    for node in reversed(nodes):
        graph.node.insert(0, node)
    onnx.checker.check_model(model)
    onnx.save(model, str(target))
    return target


def cuda_onnxruntime_available() -> bool:
    try:
        import onnxruntime as ort
    except Exception:
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


class ONNXRuntimeTextDetectionPredictor:
    """Batched PP-OCRv5 detector using ONNX Runtime CUDA/TensorRT."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        thresh: float = .30,
        box_thresh: float = .50,
        engine: str = "cuda",
        precision: str = "fp32",
        max_batch_size: int = 64,
        trt_cache_dir: str | Path | None = None,
    ):
        import onnxruntime as ort
        from paddlex.inference.models.text_detection.processors import DetResizeForTest

        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls()

        source = Path(model_path)
        self.model_path = ensure_fused_uint8_nhwc_model(source)
        self.thresh = float(thresh)
        self.box_thresh = float(box_thresh)
        self.runtime = str(engine).lower()
        self.precision = str(precision).lower()
        self.max_batch_size = int(max_batch_size)
        if self.runtime not in {"cuda", "tensorrt"}:
            raise ValueError("engine must be cuda or tensorrt")
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError("precision must be fp32 or fp16")
        if self.max_batch_size <= 0:
            raise ValueError("max_batch_size must be positive")

        self.ort = ort
        self.engine = f"onnxruntime_{self.runtime}"
        self.trt_cache_dir = Path(
            trt_cache_dir
            or (Path.home() / ".cache" / "video_text" / "ppocrv5_trt")
        ).expanduser()
        self.trt_cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = None
        self.providers = []
        self.input_name = None
        self.output_name = None
        self.profile_hw = None
        self.resize_op = DetResizeForTest(limit_side_len=960, limit_type="max")
        self.postprocess = _DefaultPostprocess(self.thresh, self.box_thresh)
        self.use_io_binding = os.environ.get("VIDEO_TEXT_ORT_IO_BINDING", "0") == "1"

        available = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(f"CUDAExecutionProvider unavailable: {available}")
        if self.runtime == "tensorrt" and "TensorrtExecutionProvider" not in available:
            raise RuntimeError(f"TensorrtExecutionProvider unavailable: {available}")
        if self.runtime == "cuda":
            self._create_cuda_session()

    def _session_options(self):
        options = self.ort.SessionOptions()
        options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return options

    @staticmethod
    def _cuda_options():
        return {
            "device_id": 0,
            "cudnn_conv_algo_search": "HEURISTIC",
            "do_copy_in_default_stream": 1,
        }

    def _create_cuda_session(self):
        providers = [
            ("CUDAExecutionProvider", self._cuda_options()),
            "CPUExecutionProvider",
        ]
        self.session = self.ort.InferenceSession(
            str(self.model_path),
            sess_options=self._session_options(),
            providers=providers,
        )
        self.providers = self.session.get_providers()
        if not self.providers or self.providers[0] != "CUDAExecutionProvider":
            raise RuntimeError(f"ONNX Runtime CUDA provider was not selected: {self.providers}")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _create_tensorrt_session(self, batch_shape):
        _, height, width, channels = [int(x) for x in batch_shape]
        if channels != 3:
            raise ValueError("TensorRT fused input must be NHWC with 3 channels")
        cache = self.trt_cache_dir / f"{height}x{width}_{self.precision}_b{self.max_batch_size}"
        cache.mkdir(parents=True, exist_ok=True)
        opt_batch = min(40, self.max_batch_size)
        input_name = "images_uint8_nhwc"
        profile = lambda n: f"{input_name}:{n}x{height}x{width}x3"
        trt_options = {
            "device_id": 0,
            "trt_fp16_enable": self.precision == "fp16",
            "trt_max_workspace_size": 2147483648,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": str(cache),
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": str(cache),
            "trt_auxiliary_streams": 0,
            "trt_builder_optimization_level": 3,
            "trt_profile_min_shapes": profile(1),
            "trt_profile_opt_shapes": profile(opt_batch),
            "trt_profile_max_shapes": profile(self.max_batch_size),
        }
        providers = [
            ("TensorrtExecutionProvider", trt_options),
            ("CUDAExecutionProvider", self._cuda_options()),
            "CPUExecutionProvider",
        ]
        self.session = self.ort.InferenceSession(
            str(self.model_path),
            sess_options=self._session_options(),
            providers=providers,
        )
        self.providers = self.session.get_providers()
        if not self.providers or self.providers[0] != "TensorrtExecutionProvider":
            raise RuntimeError(f"ONNX Runtime TensorRT provider was not selected: {self.providers}")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.profile_hw = (height, width)

    def _run_session(self, batch: np.ndarray) -> np.ndarray:
        if not self.use_io_binding:
            return self.session.run([self.output_name], {self.input_name: batch})[0]
        input_value = self.ort.OrtValue.ortvalue_from_numpy(batch, "cuda", 0)
        binding = self.session.io_binding()
        binding.bind_ortvalue_input(self.input_name, input_value)
        binding.bind_output(self.output_name, "cuda", 0)
        self.session.run_with_iobinding(binding)
        return binding.copy_outputs_to_cpu()[0]

    def predict(self, images: Sequence[np.ndarray]) -> Iterable[dict]:
        if not images:
            return iter(())
        batch, shapes, _ = prepare_fused_inputs(images, resize_op=self.resize_op)
        hw = (int(batch.shape[1]), int(batch.shape[2]))
        if self.runtime == "tensorrt":
            if self.session is None or self.profile_hw != hw:
                self.release()
                self._create_tensorrt_session(batch.shape)
        elif self.session is None:
            raise RuntimeError("ONNX Runtime session has been released")
        pred = self._run_session(batch)
        polys, scores = self.postprocess(pred, shapes, self.thresh, self.box_thresh)
        return iter([
            {"dt_polys": np.asarray(p), "dt_scores": np.asarray(s)}
            for p, s in zip(polys, scores)
        ])

    def release(self):
        self.session = None
