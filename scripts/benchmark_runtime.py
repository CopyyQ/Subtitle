from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shlex
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import torch

from scripts.compare_coordinates import compare_files, select_best_variant
from src.video_text.io_probe import bottom_roi, probe_video
from src.video_text.pipeline import PipelineConfig, SubtitlePipeline


def _csv_ints(value):
    out=[int(x.strip()) for x in str(value).split(",") if x.strip()]
    if not out or any(x<=0 for x in out):
        raise argparse.ArgumentTypeError("batches must be positive integers")
    return out


def _csv_precisions(value):
    out=[x.strip().lower() for x in str(value).split(",") if x.strip()]
    if not out or any(x not in {"fp32","fp16"} for x in out):
        raise argparse.ArgumentTypeError("precisions must contain fp32/fp16")
    return out


def _command(input_path,output,coords,args,batch,precision):
    parts=[
        sys.executable,str(ROOT/"scripts"/"run_subtitle_pipeline.py"),
        str(input_path),"--output",str(output),
        "--device",args.device,"--precision",precision,
        "--batch-size",str(batch),
        "--cpu-threads",str(args.cpu_threads),
        "--box-thickness",str(args.box_thickness),
        "--roi-bottom-fraction",str(args.roi_bottom_fraction),
        "--temporal-mode","v1",
        "--max-frames",str(args.max_frames),
        "--export-coordinates",str(coords),
        "--no-validate-chinese",
    ]
    return shlex.join(parts)


def run_variant(input_path,output_dir,args,batch,precision):
    name=f"{args.device}_{precision}_b{batch}"
    variant_dir=output_dir/name
    variant_dir.mkdir(parents=True,exist_ok=True)
    output=variant_dir/"boxed.mp4"
    coords=variant_dir/"coords.json"
    cfg=PipelineConfig(
        temporal_mode="v1",
        validate_chinese=False,
        roi_bottom_fraction=args.roi_bottom_fraction,
        device=args.device,
        precision=precision,
        batch_size=batch,
        cpu_threads=args.cpu_threads,
        box_thickness=args.box_thickness,
    )
    pipe=SubtitlePipeline(cfg)
    backend=None
    row={
        "name":name,
        "device":args.device,
        "precision":precision,
        "batch_size":batch,
        "status":"failed",
        "command":_command(input_path,output,coords,args,batch,precision),
        "output_video":str(output),
        "coordinate_json":str(coords),
    }
    try:
        info=probe_video(input_path)
        target=min(info.frame_count,args.max_frames) if args.max_frames else info.frame_count
        backend=pipe._default_backend()
        pipe.backend=backend
        y1,y2=bottom_roi(info.height,args.roi_bottom_fraction)
        warm_start=time.perf_counter()
        backend.warmup((y2-y1,info.width,3),batch_size=batch)
        row["warmup_seconds"]=time.perf_counter()-warm_start
        cache=pipe._cache_path(Path(input_path).resolve(),output.resolve())
        cache.unlink(missing_ok=True)
        result=pipe.run(
            input_path,output,max_frames=target,coordinate_json=coords
        )
        row.update(result.metrics)
        row["status"]="ok"
        row["end_to_end_fps"]=result.metrics.get("end_to_end_fps")
    except Exception as exc:
        row["error_type"]=type(exc).__name__
        row["error"]=str(exc)
    finally:
        del pipe
        if backend is not None:
            del backend
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return row


def build_parser():
    p=argparse.ArgumentParser(description="Sweep FAST V1 runtime configurations")
    p.add_argument("input")
    p.add_argument("--device",choices=["cpu","cuda"],default="cuda")
    p.add_argument("--batches",type=_csv_ints,default=_csv_ints("4,8,16,32"))
    p.add_argument("--precisions",type=_csv_precisions,default=_csv_precisions("fp32,fp16"))
    p.add_argument("--max-frames",type=int,default=600)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--cpu-threads",type=int,default=0)
    p.add_argument("--roi-bottom-fraction",type=float,default=.45)
    p.add_argument("--box-thickness",type=int,default=2)
    p.add_argument("--max-edge-delta",type=float,default=2.0)
    p.add_argument("--max-missing-frame-rate",type=float,default=0.0)
    return p


def main(argv=None):
    args=build_parser().parse_args(argv)
    if args.max_frames<0:
        raise SystemExit("--max-frames must be non-negative")
    if args.device=="cpu" and any(p=="fp16" for p in args.precisions):
        raise SystemExit("CPU sweep supports fp32 only")
    out=Path(args.output_dir).resolve()
    out.mkdir(parents=True,exist_ok=True)
    input_path=Path(args.input).resolve()

    rows=[]
    for precision in args.precisions:
        for batch in args.batches:
            row=run_variant(input_path,out,args,batch,precision)
            rows.append(row)
            print(json.dumps(row,ensure_ascii=False))

    golden_candidates=[
        r for r in rows if r.get("status")=="ok" and r.get("precision")=="fp32"
    ]
    golden=min(golden_candidates,key=lambda r:int(r["batch_size"])) if golden_candidates else None
    if golden is not None:
        for row in rows:
            if row.get("status")!="ok":
                row["comparison_passed"]=False
                continue
            comparison=compare_files(
                golden["coordinate_json"],row["coordinate_json"],
                max_edge_delta=args.max_edge_delta,
                max_missing_frame_rate=args.max_missing_frame_rate,
            )
            row["comparison"]=comparison
            row["comparison_passed"]=bool(comparison["passed"])
    else:
        for row in rows:
            row["comparison_passed"]=False

    selected=select_best_variant(rows)
    result={
        "input":str(input_path),
        "golden_variant":None if golden is None else golden["name"],
        "rows":rows,
        "selected":selected,
        "selected_command":None if selected is None else selected["command"],
    }
    result_path=out/"benchmark_results.json"
    result_path.write_text(
        json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    print(f"results={result_path}")
    if selected is not None:
        print(f"selected={selected['name']}")
        print(f"selected_command={selected['command']}")
        return 0
    return 2


if __name__=="__main__":
    raise SystemExit(main())
