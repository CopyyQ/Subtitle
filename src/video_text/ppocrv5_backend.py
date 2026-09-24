from __future__ import annotations

from typing import Any
import importlib.util
from pathlib import Path

import numpy as np
import torch

from .openvino_backend import OpenVINOTextDetectionPredictor, resolve_cpu_engine
from .onnxruntime_backend import (
    ONNXRuntimeTextDetectionPredictor,
    cuda_onnxruntime_available,
    default_ppocrv5_onnx_model,
)
from .temporal_gate import AdaptiveSubtitleGate, TemporalGateConfig
from .types import Candidate


def _onnxruntime_available() -> bool:
    return importlib.util.find_spec("onnxruntime") is not None


def _openvino_available() -> bool:
    return importlib.util.find_spec("openvino") is not None


def resolve_ppocr_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be auto, cuda, or cpu")


class PPOCRv5MobileBackend:
    model_name = "PP-OCRv5_mobile_det"

    def __init__(
        self,
        device: str | torch.device = "auto",
        batch_size: int = 16,
        high_score: float = .84,
        low_score: float = .50,
        thresh: float = .30,
        box_thresh: float = .50,
        cpu_threads: int = 0,
        cpu_engine: str = "auto",
        openvino_model: str | Path | None = None,
        openvino_num_streams: int = 0,
        openvino_fuse_preprocess: bool = False,
        gpu_engine: str = "cuda",
        trt_precision: str = "fp32",
        trt_cache_dir: str | Path | None = None,
        adaptive_gating: bool = False,
        gate_max_skip_frames: int = 2,
        gate_change_threshold: float = .02,
        gate_bright_net_threshold: float = .0075,
        predictor: Any | None = None,
    ):
        requested = device.type if isinstance(device, torch.device) else str(device)
        self.device = resolve_ppocr_device(requested)
        self.batch_size = int(batch_size)
        self.high_score = float(high_score)
        self.low_score = float(low_score)
        self.thresh = float(thresh)
        self.box_thresh = float(box_thresh)
        self.cpu_threads = int(cpu_threads)
        self.openvino_model = (
            Path(openvino_model).expanduser() if openvino_model is not None else None
        )
        self.openvino_num_streams = int(openvino_num_streams)
        self.openvino_fuse_preprocess = bool(openvino_fuse_preprocess)
        self.requested_gpu_engine = str(gpu_engine).lower()
        self.trt_precision = str(trt_precision).lower()
        self.trt_cache_dir = Path(trt_cache_dir).expanduser() if trt_cache_dir else None
        self.adaptive_gating = bool(adaptive_gating)
        self.gate_max_skip_frames = int(gate_max_skip_frames)
        self.gate_change_threshold = float(gate_change_threshold)
        self.gate_bright_net_threshold = float(gate_bright_net_threshold)
        self.cpu_engine = None
        self.gpu_engine = None
        self.gpu_acceleration_error = None
        self.precision = "fp32"
        self._gate = (
            AdaptiveSubtitleGate(
                TemporalGateConfig(
                    max_skip_frames=self.gate_max_skip_frames,
                    change_threshold=self.gate_change_threshold,
                    bright_net_threshold=self.gate_bright_net_threshold,
                    band_top_fraction=.15,
                    band_bottom_fraction=.50,
                )
            )
            if self.adaptive_gating
            else None
        )
        self._gate_frame_index = 0
        self._last_gate_row = None
        self.gate_total_frames = 0
        self.gate_inferred_frames = 0
        self.gate_skipped_frames = 0
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not 0.0 <= self.low_score <= self.high_score <= 1.0:
            raise ValueError("low_score/high_score must satisfy 0 <= low_score <= high_score <= 1")
        if not 0.0 <= self.thresh <= 1.0:
            raise ValueError("thresh must be within 0..1")
        if not 0.0 <= self.box_thresh <= 1.0:
            raise ValueError("box_thresh must be within 0..1")
        if self.cpu_threads < 0:
            raise ValueError("cpu_threads must be non-negative")
        if self.openvino_num_streams < 0:
            raise ValueError("openvino_num_streams must be non-negative")
        if self.requested_gpu_engine not in {"cuda","tensorrt"}:
            raise ValueError("gpu_engine must be cuda or tensorrt")
        if self.trt_precision not in {"fp32","fp16"}:
            raise ValueError("trt_precision must be fp32 or fp16")
        if self.gate_max_skip_frames < 0:
            raise ValueError("gate_max_skip_frames must be non-negative")
        if not 0.0 <= self.gate_change_threshold <= 1.0:
            raise ValueError("gate_change_threshold must be within 0..1")
        if not 0.0 <= self.gate_bright_net_threshold <= 1.0:
            raise ValueError("gate_bright_net_threshold must be within 0..1")

        if predictor is None and self.device.type == "cpu":
            self.cpu_engine = resolve_cpu_engine(
                cpu_engine,
                openvino_available=_openvino_available(),
                openvino_model_exists=bool(
                    self.openvino_model is not None and self.openvino_model.is_file()
                ),
                onnxruntime_available=_onnxruntime_available(),
            )
            if self.cpu_engine == "openvino":
                predictor = OpenVINOTextDetectionPredictor(
                    self.openvino_model,
                    thresh=self.thresh,
                    box_thresh=self.box_thresh,
                    cpu_threads=self.cpu_threads,
                    performance_hint="THROUGHPUT",
                    async_inference=True,
                    num_streams=self.openvino_num_streams,
                    fuse_preprocess=self.openvino_fuse_preprocess,
                )
            else:
                from paddleocr import TextDetection

                runtime_kwargs = {}
                if self.cpu_engine == "onnxruntime":
                    runtime_kwargs["engine"] = "onnxruntime"
                else:
                    # PaddlePaddle 3.3.1 + PP-OCRv5 Mobile currently fails in
                    # the default oneDNN/PIR path on this model. Fall back to
                    # the reliable plain Paddle CPU runner when optimized
                    # runtimes are not available.
                    runtime_kwargs["enable_mkldnn"] = False
                if self.cpu_threads > 0:
                    runtime_kwargs["cpu_threads"] = self.cpu_threads
                predictor = TextDetection(
                    model_name=self.model_name,
                    device="cpu",
                    enable_hpi=False,
                    thresh=self.thresh,
                    box_thresh=self.box_thresh,
                    **runtime_kwargs,
                )
        elif predictor is None:
            onnx_model = default_ppocrv5_onnx_model()
            if onnx_model is not None and cuda_onnxruntime_available():
                try:
                    predictor = ONNXRuntimeTextDetectionPredictor(
                        onnx_model,
                        thresh=self.thresh,
                        box_thresh=self.box_thresh,
                        engine=self.requested_gpu_engine,
                        precision=self.trt_precision,
                        max_batch_size=self.batch_size,
                        trt_cache_dir=self.trt_cache_dir,
                    )
                    self.gpu_engine = getattr(predictor, "engine", f"onnxruntime_{self.requested_gpu_engine}")
                    self.precision = getattr(predictor, "precision", "fp32")
                except Exception as exc:
                    self.gpu_acceleration_error = f"{type(exc).__name__}: {exc}"
                    predictor = None
                if predictor is None and self.requested_gpu_engine == "tensorrt":
                    try:
                        predictor = ONNXRuntimeTextDetectionPredictor(
                            onnx_model,
                            thresh=self.thresh,
                            box_thresh=self.box_thresh,
                            engine="cuda",
                            precision="fp32",
                            max_batch_size=self.batch_size,
                        )
                        self.gpu_engine = "onnxruntime_cuda"
                        self.precision = "fp32"
                    except Exception as fallback_exc:
                        detail = f"{type(fallback_exc).__name__}: {fallback_exc}"
                        self.gpu_acceleration_error = f"{self.gpu_acceleration_error}; CUDA fallback: {detail}"
                        predictor = None
            if predictor is None:
                from paddleocr import TextDetection

                predictor = TextDetection(
                    model_name=self.model_name,
                    device="gpu:0",
                    enable_hpi=False,
                    thresh=self.thresh,
                    box_thresh=self.box_thresh,
                )
                self.gpu_engine = "paddle"
        self.predictor = predictor
        self.gpu_providers = getattr(predictor, "providers", None)

    def release_accelerator(self):
        release = getattr(self.predictor, "release", None)
        if callable(release):
            release()

    @staticmethod
    def _clone_predictor_row(row):
        return {
            "dt_polys": [
                np.asarray(poly).copy()
                for poly in row["dt_polys"]
            ],
            "dt_scores": np.asarray(row["dt_scores"]).copy(),
        }

    def reset_temporal_gate(self) -> None:
        if self._gate is not None:
            self._gate.reset()
        self._gate_frame_index = 0
        self._last_gate_row = None
        self.gate_total_frames = 0
        self.gate_inferred_frames = 0
        self.gate_skipped_frames = 0

    def _predict_rows(self, images: list[np.ndarray]):
        if self._gate is None:
            return list(self.predictor.predict(images))

        detect_flags = []
        selected = []
        for offset, image in enumerate(images):
            should_detect = self._gate.should_detect(
                image,
                self._gate_frame_index + offset,
            )
            # The first output of a stream must always come from the model.
            if self._last_gate_row is None and not detect_flags:
                should_detect = True
            detect_flags.append(bool(should_detect))
            if should_detect:
                selected.append(image)

        detected_rows = list(self.predictor.predict(selected))
        if len(detected_rows) != len(selected):
            raise RuntimeError(
                f"detector batch length mismatch: {len(detected_rows)} results "
                f"for {len(selected)} gated images"
            )

        detected_iter = iter(detected_rows)
        rows = []
        last = self._last_gate_row
        inferred = 0
        for should_detect in detect_flags:
            if should_detect:
                last = self._clone_predictor_row(next(detected_iter))
                inferred += 1
            elif last is None:
                raise RuntimeError("adaptive gate attempted reuse before first detection")
            rows.append(self._clone_predictor_row(last))

        self._last_gate_row = self._clone_predictor_row(last)
        self._gate_frame_index += len(images)
        self.gate_total_frames += len(images)
        self.gate_inferred_frames += inferred
        self.gate_skipped_frames += len(images) - inferred
        return rows

    @staticmethod
    def _candidate(poly, score: float, level: str) -> Candidate:
        points = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
        if len(points) < 2:
            raise ValueError("text polygon must contain at least two points")
        bbox = np.array(
            [
                float(points[:, 0].min()),
                float(points[:, 1].min()),
                float(points[:, 0].max()),
                float(points[:, 1].max()),
            ],
            dtype=np.float32,
        )
        return Candidate(bbox, float(score), level, "ppocrv5_mobile")

    def detect_batch(
        self, images: list[np.ndarray]
    ) -> list[tuple[list[Candidate], list[Candidate]]]:
        rows = self._predict_rows(images)
        if len(rows) != len(images):
            raise RuntimeError(
                f"detector batch length mismatch: {len(rows)} results for {len(images)} images"
            )
        output = []
        for row in rows:
            polys = row["dt_polys"]
            scores = row["dt_scores"]
            high: list[Candidate] = []
            low: list[Candidate] = []
            for poly, raw_score in zip(polys, scores):
                score = float(raw_score)
                if score < self.low_score:
                    continue
                if score >= self.high_score:
                    high.append(self._candidate(poly, score, "HIGH"))
                else:
                    low.append(self._candidate(poly, score, "LOW"))
            output.append((high, low))
        return output

    def warmup(self, shape, batch_size: int | None = None) -> None:
        batch = int(batch_size or self.batch_size)
        h, w = int(shape[0]), int(shape[1])
        images = [np.zeros((h, w, 3), np.uint8) for _ in range(batch)]
        self.detect_batch(images)
