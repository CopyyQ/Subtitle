#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))


def export_fp32_ir(
    onnx_path: str | Path,
    output_xml: str | Path,
    *,
    ov_module=None,
) -> Path:
    onnx_path = Path(onnx_path)
    output_xml = Path(output_xml)
    if not onnx_path.is_file():
        raise FileNotFoundError(onnx_path)
    output_xml.parent.mkdir(parents=True, exist_ok=True)

    if ov_module is None:
        import openvino as ov
        ov_module = ov

    model = ov_module.convert_model(str(onnx_path))
    ov_module.save_model(model, str(output_xml), compress_to_fp16=False)
    return output_xml


def build_calibration_frame_indices(
    total_frames: int,
    max_samples: int,
    forced_frames=(),
) -> list[int]:
    total_frames = int(total_frames)
    max_samples = int(max_samples)
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    forced = sorted({
        int(x)
        for x in forced_frames
        if 0 <= int(x) < total_frames
    })
    if len(forced) > max_samples:
        raise ValueError("forced calibration frames exceed max_samples")

    remaining = max_samples - len(forced)
    selected = set(forced)
    if remaining > 0:
        candidates = np.linspace(
            0,
            total_frames - 1,
            num=max(remaining * 4, remaining),
            dtype=int,
        )
        for idx in candidates:
            selected.add(int(idx))
            if len(selected) >= max_samples:
                break

    if len(selected) < max_samples:
        for idx in range(total_frames):
            selected.add(idx)
            if len(selected) >= max_samples:
                break

    return sorted(selected)


def quantize_int8_ir(
    fp32_xml: str | Path,
    output_xml: str | Path,
    calibration_inputs,
    *,
    nncf_module=None,
    ov_module=None,
) -> Path:
    fp32_xml = Path(fp32_xml)
    output_xml = Path(output_xml)
    if not fp32_xml.is_file():
        raise FileNotFoundError(fp32_xml)

    samples = list(calibration_inputs)
    if not samples:
        raise ValueError("calibration_inputs must not be empty")

    if ov_module is None:
        import openvino as ov
        ov_module = ov
    if nncf_module is None:
        import nncf
        nncf_module = nncf

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    core = ov_module.Core()
    model = core.read_model(str(fp32_xml))
    dataset = nncf_module.Dataset(samples)
    quantized = nncf_module.quantize(
        model,
        dataset,
        subset_size=len(samples),
        target_device=nncf_module.TargetDevice.CPU,
    )
    ov_module.save_model(
        quantized,
        str(output_xml),
        compress_to_fp16=False,
    )
    return output_xml


def collect_calibration_inputs(
    video_path: str | Path,
    frame_indices,
    *,
    roi_bottom_fraction: float = .45,
):
    import cv2

    from src.video_text.openvino_backend import _default_preprocess

    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    y0 = int(round(height * (1.0 - float(roi_bottom_fraction))))
    samples = []
    try:
        for frame_index in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = cap.read()
            if not ok:
                continue
            roi = frame[y0:].copy()
            batch, _ = _default_preprocess([roi])
            samples.append(batch)
    finally:
        cap.release()

    if not samples:
        raise RuntimeError("no calibration frames could be decoded")
    return samples


def build_parser():
    p = argparse.ArgumentParser(
        description="Export PP-OCRv5 Mobile ONNX detector to OpenVINO IR."
    )
    p.add_argument("--onnx", required=True)
    p.add_argument("--output", required=True, help="Output FP32 .xml path")
    p.add_argument("--quantize-int8-output")
    p.add_argument("--calibration-video")
    p.add_argument("--max-samples", type=int, default=256)
    p.add_argument("--force-frame", action="append", type=int, default=[])
    p.add_argument("--roi-bottom-fraction", type=float, default=.45)
    return p


def main():
    args = build_parser().parse_args()
    fp32_out = export_fp32_ir(args.onnx, args.output)
    print(f"fp32={fp32_out}")

    if args.quantize_int8_output:
        if not args.calibration_video:
            raise SystemExit(
                "--calibration-video is required with --quantize-int8-output"
            )
        import cv2

        cap = cv2.VideoCapture(str(args.calibration_video))
        if not cap.isOpened():
            raise SystemExit(f"cannot open calibration video: {args.calibration_video}")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        indices = build_calibration_frame_indices(
            total_frames,
            args.max_samples,
            args.force_frame,
        )
        samples = collect_calibration_inputs(
            args.calibration_video,
            indices,
            roi_bottom_fraction=args.roi_bottom_fraction,
        )
        int8_out = quantize_int8_ir(
            fp32_out,
            args.quantize_int8_output,
            samples,
        )
        print(f"int8={int8_out}")
        print(f"calibration_samples={len(samples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
