from __future__ import annotations

from typing import Any
import importlib.util
from pathlib import Path

import numpy as np
import torch

from .openvino_backend import OpenVINOTextDetectionPredictor, resolve_cpu_engine
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
        self.cpu_engine = None
        self.precision = "fp32"
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
            from paddleocr import TextDetection

            predictor = TextDetection(
                model_name=self.model_name,
                device="gpu:0",
                enable_hpi=False,
                thresh=self.thresh,
                box_thresh=self.box_thresh,
            )
        self.predictor = predictor

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
        rows = list(self.predictor.predict(images))
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
