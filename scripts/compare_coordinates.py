from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _key(row):
    return (
        int(row["frame"]),
        int(row.get("subtitle_id",row.get("track_id",0))),
        int(row.get("line_id",0)),
    )


def _record_map(payload):
    return {_key(row):row for row in payload.get("records",[])}


def _quality_regressions(golden,candidate,max_edge_delta):
    gm=golden.get("metadata",{})
    cm=candidate.get("metadata",{})
    regressions={}
    if (
        "multiline_overlap_frame_count" in gm
        and "multiline_overlap_frame_count" in cm
        and float(cm["multiline_overlap_frame_count"])
            > float(gm["multiline_overlap_frame_count"])
    ):
        regressions["multiline_overlap_frame_count"]={
            "golden":gm["multiline_overlap_frame_count"],
            "candidate":cm["multiline_overlap_frame_count"],
        }
    for key in (
        "stabilized_edge_motion_median_px",
        "stabilized_edge_motion_p95_px",
        "stabilized_corner_motion_median_px",
        "stabilized_corner_motion_p95_px",
    ):
        if key in gm and key in cm:
            if float(cm[key])>float(gm[key])+float(max_edge_delta):
                regressions[key]={"golden":gm[key],"candidate":cm[key]}
    key="bbox_excess_area_estimate"
    if key in gm and key in cm:
        allowed=float(gm[key])*1.05+1e-6
        if float(cm[key])>allowed:
            regressions[key]={"golden":gm[key],"candidate":cm[key]}
    return regressions


def compare_payloads(
    golden,
    candidate,
    *,
    max_edge_delta=2.0,
    max_missing_frame_rate=0.0,
):
    gmap=_record_map(golden)
    cmap=_record_map(candidate)
    gkeys=set(gmap)
    ckeys=set(cmap)
    missing=sorted(gkeys-ckeys)
    extra=sorted(ckeys-gkeys)
    matched=sorted(gkeys&ckeys)

    edge_deltas=[]
    for key in matched:
        gb=np.asarray(gmap[key]["bbox"],dtype=np.float64)
        cb=np.asarray(cmap[key]["bbox"],dtype=np.float64)
        edge_deltas.extend(np.abs(gb-cb).tolist())

    gframes={k[0] for k in gkeys}
    cframes={k[0] for k in ckeys}
    missing_frames=sorted(gframes-cframes)
    ghost_frames=sorted(cframes-gframes)
    missing_rate=len(missing)/max(1,len(gkeys))
    if edge_deltas:
        arr=np.asarray(edge_deltas,dtype=np.float64)
        mean_delta=float(np.mean(arr))
        p95_delta=float(np.percentile(arr,95))
        max_delta=float(np.max(arr))
    else:
        mean_delta=p95_delta=max_delta=0.0

    quality=_quality_regressions(golden,candidate,max_edge_delta)
    passed=(
        missing_rate<=float(max_missing_frame_rate)
        and len(extra)==0
        and max_delta<=float(max_edge_delta)
        and not quality
    )
    return {
        "passed":bool(passed),
        "matched_box_count":len(matched),
        "missing_key_count":len(missing),
        "extra_key_count":len(extra),
        "missing_frame_rate":float(missing_rate),
        "mean_edge_delta_px":mean_delta,
        "p95_edge_delta_px":p95_delta,
        "max_edge_delta_px":max_delta,
        "boxed_frame_sets_equal":gframes==cframes,
        "missing_boxed_frames":missing_frames,
        "ghost_boxed_frames":ghost_frames,
        "quality_regressions":quality,
    }


def compare_files(golden_path,candidate_path,**kwargs):
    golden=json.loads(Path(golden_path).read_text(encoding="utf-8"))
    candidate=json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    return compare_payloads(golden,candidate,**kwargs)


def select_best_variant(rows):
    eligible=[
        r for r in rows
        if r.get("status")=="ok"
        and r.get("comparison_passed") is True
        and r.get("end_to_end_fps") is not None
    ]
    if not eligible:
        return None
    fastest=max(float(r["end_to_end_fps"]) for r in eligible)
    contenders=[
        r for r in eligible
        if float(r["end_to_end_fps"])>=fastest*.97
    ]
    return min(
        contenders,
        key=lambda r:(
            int(r.get("batch_size",10**9)),
            0 if str(r.get("precision"))=="fp32" else 1,
            -float(r["end_to_end_fps"]),
        ),
    )


def build_parser():
    p=argparse.ArgumentParser(description="Compare subtitle bbox coordinate JSON")
    p.add_argument("golden")
    p.add_argument("candidate")
    p.add_argument("--max-edge-delta",type=float,default=2.0)
    p.add_argument("--max-missing-frame-rate",type=float,default=0.0)
    return p


def main(argv=None):
    args=build_parser().parse_args(argv)
    result=compare_files(
        args.golden,args.candidate,
        max_edge_delta=args.max_edge_delta,
        max_missing_frame_rate=args.max_missing_frame_rate,
    )
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result["passed"] else 1


if __name__=="__main__":
    raise SystemExit(main())
