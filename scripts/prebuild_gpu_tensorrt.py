from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import cv2

from src.video_text.onnxruntime_backend import ONNXRuntimeTextDetectionPredictor


def build_parser():
    p=argparse.ArgumentParser(description="Prebuild cached TensorRT engine for PP-OCRv5 GPU runtime")
    p.add_argument("video")
    p.add_argument("--model",required=True)
    p.add_argument("--cache-dir",required=True)
    p.add_argument("--batch-size",type=int,default=64)
    p.add_argument("--precision",choices=["fp32","fp16"],default="fp32")
    p.add_argument("--roi-bottom-fraction",type=float,default=.45)
    return p

def main():
    args=build_parser().parse_args()
    cap=cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    frames=[]
    target=min(40,args.batch_size)
    while len(frames)<target:
        ok,frame=cap.read()
        if not ok: break
        y0=int(round(frame.shape[0]*(1.0-args.roi_bottom_fraction)))
        frames.append(frame[y0:].copy())
    cap.release()
    if not frames:
        raise SystemExit("no frames decoded for TensorRT prebuild")

    predictor=ONNXRuntimeTextDetectionPredictor(
        args.model,engine="tensorrt",precision=args.precision,
        max_batch_size=args.batch_size,trt_cache_dir=args.cache_dir,
    )
    start=time.perf_counter()
    list(predictor.predict(frames))
    elapsed=time.perf_counter()-start
    print(f"tensorrt_prebuild_seconds={elapsed:.3f}")
    print(f"tensorrt_cache_dir={Path(args.cache_dir).expanduser().resolve()}")
    print(f"providers={predictor.providers}")
    predictor.release()
    return 0


if __name__=="__main__":
    raise SystemExit(main())
