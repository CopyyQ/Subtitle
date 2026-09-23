from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from src.video_text.io_probe import bottom_roi, probe_video
from src.video_text.pipeline import PipelineConfig, SubtitlePipeline

def roi_fraction(value):
    v=float(value)
    if not .25<=v<=.45:
        raise argparse.ArgumentTypeError("ROI bottom fraction must be within 0.25..0.45")
    return v

def build_parser():
    p=argparse.ArgumentParser(description="Detect and stabilize hard-coded Chinese subtitles")
    p.add_argument("input")
    p.add_argument("--output")
    p.add_argument("--roi-bottom-fraction",type=roi_fraction,default=.45)
    p.add_argument("--codec",choices=["h264","h265"],default="h264")
    p.add_argument("--detector",choices=["ppocrv5_mobile","fast"],default="ppocrv5_mobile")
    p.add_argument("--high-score",type=float,default=.84)
    p.add_argument("--low-score",type=float,default=.50)
    p.add_argument("--ppocr-thresh",type=float,default=.30)
    p.add_argument("--ppocr-box-thresh",type=float,default=.50)
    p.add_argument("--high-min-area",type=int,default=250)
    p.add_argument("--low-min-area",type=int,default=30)
    p.add_argument("--max-internal-gap",type=int,choices=[1,2],default=2)
    p.add_argument("--smoothing-window",type=int,default=5)
    p.add_argument("--outline-pad-ratio",type=float,default=.08)
    p.add_argument("--min-outline-pad",type=int,default=3)
    p.add_argument("--temporal-mode",choices=["v4","v5","v5_5","v1"],default="v1")
    p.add_argument("--device",choices=["auto","cpu","cuda"],default="auto")
    p.add_argument("--precision",choices=["auto","fp32","fp16"],default="auto")
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--decode-prefetch-batches",type=int,default=4)
    p.add_argument("--cpu-threads",type=int,default=0)
    p.add_argument("--box-thickness",type=int,default=2)
    p.add_argument("--max-frames",type=int,default=0)
    p.add_argument("--export-coordinates")
    p.add_argument("--validate-chinese",dest="validate_chinese",action="store_true",default=False)
    p.add_argument("--no-validate-chinese",dest="validate_chinese",action="store_false")
    p.add_argument("--recognize-text",action="store_true")
    p.add_argument("--export-srt")
    p.add_argument("--dry-probe",action="store_true")
    return p

def main(argv=None):
    p=build_parser()
    args=p.parse_args(argv)
    if args.export_srt and not args.recognize_text:
        p.error("--export-srt requires --recognize-text")
    if args.export_srt and args.recognize_text:
        p.error("SRT transcription is not enabled in detection-only mode; use --export-coordinates JSON")
    info=probe_video(args.input)
    y1,y2=bottom_roi(info.height,args.roi_bottom_fraction)
    if args.dry_probe:
        print(json.dumps({
            "input":str(Path(args.input).resolve()),
            "width":info.width,"height":info.height,"fps":info.fps,
            "frame_count":info.frame_count,"fourcc":info.fourcc,
            "roi":[0,y1,info.width,y2],
        },indent=2))
        return 0
    if not args.output:
        p.error("--output is required unless --dry-probe is used")
    cfg=PipelineConfig(
        roi_bottom_fraction=args.roi_bottom_fraction,
        detector=args.detector,
        high_score=args.high_score,low_score=args.low_score,
        ppocr_thresh=args.ppocr_thresh,ppocr_box_thresh=args.ppocr_box_thresh,
        high_min_area=args.high_min_area,low_min_area=args.low_min_area,
        max_internal_gap=args.max_internal_gap,
        smoothing_window=args.smoothing_window,
        output_codec=args.codec,validate_chinese=args.validate_chinese,
        export_srt=bool(args.export_srt),
        outline_pad_ratio=args.outline_pad_ratio,
        min_outline_pad=args.min_outline_pad,
        temporal_mode=args.temporal_mode,
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        decode_prefetch_batches=args.decode_prefetch_batches,
        cpu_threads=args.cpu_threads,
        box_thickness=args.box_thickness,
    )
    result=SubtitlePipeline(cfg).run(
        args.input,args.output,max_frames=args.max_frames,
        coordinate_json=args.export_coordinates,
    )
    print(json.dumps(result.metrics,ensure_ascii=False,indent=2))
    print(f"video={result.output_video}")
    print(f"coordinates={result.coordinate_json}")
    print(f"review_dir={result.review_dir}")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
