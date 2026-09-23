#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import queue
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple

import numpy as np


class Variant(NamedTuple):
    name: str
    batch_size: int
    threaded: bool = False
    enable_hpi: bool = False
    engine: str | None = "paddle_static"
    use_tensorrt: bool = False
    precision: str = "fp32"
    dual: bool = False


def build_variants(
    batches=(8, 16, 24, 32),
    include_hpi=True,
    include_onnx=True,
    include_trt=True,
    include_dual=True,
):
    batches=tuple(int(x) for x in batches)
    rows=[Variant(f"static_b{b}",b) for b in batches]
    for b in (16,24,32):
        if b in batches:
            rows.append(Variant(f"threaded_b{b}",b,threaded=True))
    anchor=16 if 16 in batches else batches[0]
    if include_hpi:
        rows.append(Variant(f"hpi_b{anchor}",anchor,enable_hpi=True,engine=None))
    if include_onnx:
        rows.append(Variant(f"onnx_b{anchor}",anchor,engine="onnxruntime"))
    if include_trt:
        rows.append(Variant(
            f"trt_fp16_b{anchor}",anchor,
            engine="paddle_static",use_tensorrt=True,precision="fp16",
        ))
    if include_dual:
        rows.append(Variant(f"dual_static_b{anchor}",anchor,dual=True))
    return rows


def select_best_variant(rows):
    good=[
        r for r in rows
        if r.get("quality_pass") is True
        and not r.get("error")
        and math.isfinite(float(r.get("wall_fps",0.0)))
    ]
    if not good:
        return None
    return max(good,key=lambda r:float(r["wall_fps"]))


def _center(box):
    b=np.asarray(box,dtype=np.float64)
    return (b[:2]+b[2:])/2.0


def center_quality(reference,candidate,threshold_px=25.0):
    reference={int(k):v for k,v in reference.items()}
    candidate={int(k):v for k,v in candidate.items()}
    ref_boxes=sum(len(v) for v in reference.values())
    cand_boxes=sum(len(v) for v in candidate.values())
    matched=0
    distances=[]
    for frame in sorted(set(reference)|set(candidate)):
        refs=[np.asarray(x,dtype=np.float64) for x in reference.get(frame,[])]
        cands=[np.asarray(x,dtype=np.float64) for x in candidate.get(frame,[])]
        pairs=[]
        for i,a in enumerate(refs):
            ca=_center(a)
            for j,b in enumerate(cands):
                pairs.append((float(np.linalg.norm(ca-_center(b))),i,j))
        used_r=set(); used_c=set()
        for dist,i,j in sorted(pairs):
            if dist>threshold_px:
                break
            if i in used_r or j in used_c:
                continue
            used_r.add(i); used_c.add(j)
            matched+=1; distances.append(dist)
    ref_frames={f for f,v in reference.items() if v}
    cand_frames={f for f,v in candidate.items() if v}
    missing=len(ref_frames-cand_frames)
    ghost=len(cand_frames-ref_frames)
    recall=matched/max(ref_boxes,1)
    precision=matched/max(cand_boxes,1)
    return {
        "matched":matched,
        "reference_boxes":ref_boxes,
        "candidate_boxes":cand_boxes,
        "precision_center":precision,
        "recall_center":recall,
        "f1_center":(2*matched/max(ref_boxes+cand_boxes,1)),
        "center_mean_px":float(np.mean(distances)) if distances else None,
        "center_p95_px":float(np.percentile(distances,95)) if distances else None,
        "missing_boxed_frames":missing,
        "ghost_boxed_frames":ghost,
        "quality_pass":(
            missing==0 and ghost==0 and
            recall>=.995 and precision>=.995
        ),
    }


def _result_boxes(result,y0,score_floor=.50):
    polys=np.asarray(result["dt_polys"],dtype=object)
    scores=np.asarray(result["dt_scores"],dtype=np.float64)
    boxes=[]
    for poly,score in zip(polys,scores):
        if float(score)<score_floor:
            continue
        p=np.asarray(poly,dtype=np.float64).reshape(-1,2)
        if len(p)<2:
            continue
        boxes.append([
            float(p[:,0].min()),
            float(p[:,1].min()+y0),
            float(p[:,0].max()),
            float(p[:,1].max()+y0),
        ])
    return boxes


def _load_golden(path,start_frame,max_frames):
    if not path:
        return {}
    data=json.loads(Path(path).read_text(encoding="utf-8"))
    end=start_frame+max_frames
    out={}
    for row in data.get("records",[]):
        frame=int(row["frame"])
        if start_frame<=frame<end:
            out.setdefault(frame,[]).append(row["bbox"])
    return out


def _gpu_snapshot():
    try:
        p=subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,text=True,timeout=2,check=False,
        )
        line=p.stdout.strip().splitlines()[0]
        util,mem=[float(x.strip()) for x in line.split(",")[:2]]
        return util,mem
    except Exception:
        return None


class GPUSampler:
    def __init__(self,interval=.25):
        self.interval=float(interval)
        self.stop_evt=threading.Event()
        self.samples=[]
        self.thread=None

    def __enter__(self):
        def run():
            while not self.stop_evt.is_set():
                v=_gpu_snapshot()
                if v is not None:
                    self.samples.append(v)
                self.stop_evt.wait(self.interval)
        self.thread=threading.Thread(target=run,daemon=True)
        self.thread.start()
        return self

    def __exit__(self,*exc):
        self.stop_evt.set()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def summary(self):
        if not self.samples:
            return {}
        util=np.asarray([x[0] for x in self.samples],dtype=np.float64)
        mem=np.asarray([x[1] for x in self.samples],dtype=np.float64)
        return {
            "gpu_util_mean":float(util.mean()),
            "gpu_util_p95":float(np.percentile(util,95)),
            "gpu_util_max":float(util.max()),
            "gpu_mem_mean_mb":float(mem.mean()),
            "gpu_mem_max_mb":float(mem.max()),
        }


def make_predictor(variant,device,thresh,box_thresh):
    from paddleocr import TextDetection
    kwargs={
        "model_name":"PP-OCRv5_mobile_det",
        "device":device,
        "enable_hpi":variant.enable_hpi,
        "thresh":float(thresh),
        "box_thresh":float(box_thresh),
    }
    if variant.engine is not None:
        kwargs["engine"]=variant.engine
    if variant.use_tensorrt:
        kwargs["use_tensorrt"]=True
        kwargs["precision"]=variant.precision
    return TextDetection(**kwargs)


def _read_batch(cap,count,y0):
    frames=[]
    ids=[]
    t0=time.perf_counter()
    for _ in range(count):
        ok,frame=cap.read()
        if not ok:
            break
        ids.append(int(cap.get(1))-1)
        frames.append(frame[y0:].copy())
    return ids,frames,time.perf_counter()-t0


def _warmup(predictor,video,start_frame,y0,count=8):
    import cv2
    cap=cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(start_frame))
    _,frames,_=_read_batch(cap,int(count),y0)
    cap.release()
    if frames:
        list(predictor.predict(frames))


def _sequential_run(predictor,video,start_frame,max_frames,batch_size,y0,score_floor):
    import cv2
    cap=cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(start_frame))
    boxes={}
    decode_s=0.0; infer_s=0.0; done=0
    t0=time.perf_counter()
    while done<max_frames:
        want=min(batch_size,max_frames-done)
        ids,frames,dt=_read_batch(cap,want,y0)
        decode_s+=dt
        if not frames:
            break
        ti=time.perf_counter()
        results=list(predictor.predict(frames))
        infer_s+=time.perf_counter()-ti
        for fid,res in zip(ids,results):
            boxes[int(fid)]=_result_boxes(res,y0,score_floor)
        done+=len(frames)
    wall=time.perf_counter()-t0
    cap.release()
    return boxes,done,decode_s,infer_s,wall


def _threaded_run(predictor,video,start_frame,max_frames,batch_size,y0,score_floor):
    import cv2
    q=queue.Queue(maxsize=4)
    sentinel=object()
    stats={"decode_s":0.0}
    def producer():
        cap=cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES,int(start_frame))
        done=0
        try:
            while done<max_frames:
                want=min(batch_size,max_frames-done)
                ids,frames,dt=_read_batch(cap,want,y0)
                stats["decode_s"]+=dt
                if not frames:
                    break
                q.put((ids,frames))
                done+=len(frames)
        finally:
            cap.release()
            q.put(sentinel)
    boxes={};infer_s=0.0;done=0
    t0=time.perf_counter()
    th=threading.Thread(target=producer,daemon=True)
    th.start()
    while True:
        item=q.get()
        if item is sentinel:
            break
        ids,frames=item
        ti=time.perf_counter()
        results=list(predictor.predict(frames))
        infer_s+=time.perf_counter()-ti
        for fid,res in zip(ids,results):
            boxes[int(fid)]=_result_boxes(res,y0,score_floor)
        done+=len(frames)
    th.join()
    wall=time.perf_counter()-t0
    return boxes,done,stats["decode_s"],infer_s,wall


def _dual_run(predictors,video,start_frame,max_frames,batch_size,y0,score_floor):
    import cv2
    cap=cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES,int(start_frame))
    batches=[]
    decode_s=0.0; done=0
    t_decode=time.perf_counter()
    while done<max_frames:
        want=min(batch_size,max_frames-done)
        ids,frames,dt=_read_batch(cap,want,y0)
        decode_s+=dt
        if not frames:
            break
        batches.append((ids,frames))
        done+=len(frames)
    cap.release()
    decode_wall=time.perf_counter()-t_decode

    def worker(args):
        idx,ids,frames=args
        pred=predictors[idx%len(predictors)]
        t=time.perf_counter()
        res=list(pred.predict(frames))
        return ids,res,time.perf_counter()-t

    boxes={};infer_sum=0.0
    t0=time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures=[
            ex.submit(worker,(i,ids,frames))
            for i,(ids,frames) in enumerate(batches)
        ]
        for fut in futures:
            ids,results,dt=fut.result()
            infer_sum+=dt
            for fid,res in zip(ids,results):
                boxes[int(fid)]=_result_boxes(res,y0,score_floor)
    infer_wall=time.perf_counter()-t0
    # decode is intentionally preloaded for a conservative dual-predictor probe
    wall=decode_wall+infer_wall
    return boxes,done,decode_s,infer_sum,wall


def benchmark_variant(
    variant,video,start_frame,max_frames,roi_bottom_fraction,
    device,thresh,box_thresh,score_floor,
):
    import cv2
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video}")
    h=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    y0=int(round(h*(1.0-float(roi_bottom_fraction))))
    t_load=time.perf_counter()
    predictor=make_predictor(variant,device,thresh,box_thresh)
    predictors=[predictor]
    if variant.dual:
        predictors.append(make_predictor(variant,device,thresh,box_thresh))
    load_s=time.perf_counter()-t_load
    _warmup(predictor,video,start_frame,y0,min(8,variant.batch_size))
    if variant.dual:
        _warmup(predictors[1],video,start_frame,y0,min(8,variant.batch_size))
    with GPUSampler() as sampler:
        if variant.dual:
            boxes,done,decode_s,infer_s,wall=_dual_run(
                predictors,video,start_frame,max_frames,
                variant.batch_size,y0,score_floor,
            )
        elif variant.threaded:
            boxes,done,decode_s,infer_s,wall=_threaded_run(
                predictor,video,start_frame,max_frames,
                variant.batch_size,y0,score_floor,
            )
        else:
            boxes,done,decode_s,infer_s,wall=_sequential_run(
                predictor,video,start_frame,max_frames,
                variant.batch_size,y0,score_floor,
            )
    row={
        "name":variant.name,
        "batch_size":variant.batch_size,
        "threaded":variant.threaded,
        "enable_hpi":variant.enable_hpi,
        "engine":variant.engine,
        "use_tensorrt":variant.use_tensorrt,
        "precision":variant.precision,
        "dual":variant.dual,
        "frames":done,
        "model_load_seconds":load_s,
        "decode_seconds":decode_s,
        "inference_seconds_sum":infer_s,
        "wall_seconds":wall,
        "wall_fps":done/max(wall,1e-9),
        "roi_y0":y0,
        "roi_width":w,
        "roi_height":h-y0,
        "boxes":boxes,
    }
    row.update(sampler.summary())
    return row


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--golden-json")
    ap.add_argument("--output",default="outputs/ppocr_performance_gate.json")
    ap.add_argument("--start-frame",type=int,default=1600)
    ap.add_argument("--max-frames",type=int,default=300)
    ap.add_argument("--roi-bottom-fraction",type=float,default=.45)
    ap.add_argument("--device",default="gpu:0")
    ap.add_argument("--batches",default="8,16,24,32")
    ap.add_argument("--thresh",type=float,default=.30)
    ap.add_argument("--box-thresh",type=float,default=.50)
    ap.add_argument("--score-floor",type=float,default=.50)
    ap.add_argument("--skip-hpi",action="store_true")
    ap.add_argument("--skip-onnx",action="store_true")
    ap.add_argument("--skip-trt",action="store_true")
    ap.add_argument("--skip-dual",action="store_true")
    args=ap.parse_args()

    batches=tuple(int(x) for x in args.batches.split(",") if x.strip())
    variants=build_variants(
        batches,
        include_hpi=not args.skip_hpi,
        include_onnx=not args.skip_onnx,
        include_trt=not args.skip_trt,
        include_dual=not args.skip_dual,
    )
    rows=[]
    for v in variants:
        print(f"PERF_GATE_START {v.name}",flush=True)
        try:
            row=benchmark_variant(
                v,args.video,args.start_frame,args.max_frames,
                args.roi_bottom_fraction,args.device,
                args.thresh,args.box_thresh,args.score_floor,
            )
            print(
                f"PERF_GATE_DONE {v.name} "
                f"fps={row['wall_fps']:.2f} "
                f"gpu_mean={row.get('gpu_util_mean','na')}",
                flush=True,
            )
        except Exception as exc:
            row={
                "name":v.name,
                "batch_size":v.batch_size,
                "threaded":v.threaded,
                "enable_hpi":v.enable_hpi,
                "engine":v.engine,
                "use_tensorrt":v.use_tensorrt,
                "precision":v.precision,
                "dual":v.dual,
                "error":f"{type(exc).__name__}: {exc}",
                "wall_fps":0.0,
                "boxes":{},
            }
            print(f"PERF_GATE_ERROR {v.name} {row['error']}",flush=True)
        rows.append(row)

    baseline=next((r for r in rows if r["name"]=="static_b16" and not r.get("error")),None)
    if baseline is None:
        baseline=next((r for r in rows if not r.get("error")),None)
    baseline_boxes=baseline["boxes"] if baseline else {}
    golden=_load_golden(args.golden_json,args.start_frame,args.max_frames)
    for row in rows:
        if row.get("error"):
            row["quality_pass"]=False
            continue
        q=center_quality(baseline_boxes,row["boxes"],threshold_px=3.0)
        row["quality_vs_baseline"]=q
        row["quality_pass"]=q["quality_pass"]
        if golden:
            row["quality_vs_golden"]=center_quality(golden,row["boxes"],threshold_px=25.0)
        row.pop("boxes",None)

    best=select_best_variant(rows)
    payload={
        "video":str(Path(args.video).resolve()),
        "start_frame":args.start_frame,
        "max_frames":args.max_frames,
        "roi_bottom_fraction":args.roi_bottom_fraction,
        "device":args.device,
        "thresh":args.thresh,
        "box_thresh":args.box_thresh,
        "score_floor":args.score_floor,
        "baseline_variant":baseline["name"] if baseline else None,
        "best_variant":best["name"] if best else None,
        "results":rows,
    }
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8")
    print("PERF_GATE_SUMMARY",json.dumps({
        "best_variant":payload["best_variant"],
        "results":[
            {
                "name":r["name"],
                "fps":round(float(r.get("wall_fps",0.0)),2),
                "quality_pass":r.get("quality_pass",False),
                "gpu_mean":r.get("gpu_util_mean"),
                "error":r.get("error"),
            }
            for r in rows
        ],
    },ensure_ascii=False),flush=True)
    return 0 if best else 2


if __name__=="__main__":
    raise SystemExit(main())
