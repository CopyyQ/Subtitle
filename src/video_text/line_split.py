from __future__ import annotations

import cv2
import numpy as np

from .types import Candidate


def _active_runs(active):
    runs=[]
    start=None
    values=active.tolist()+[False]
    for i,value in enumerate(values):
        if value and start is None:
            start=i
        elif not value and start is not None:
            runs.append([start,i-1])
            start=None
    return runs


def _merge_close_runs(runs,max_gap=2):
    if not runs:
        return []
    merged=[runs[0][:]]
    for start,end in runs[1:]:
        if start-merged[-1][1]-1 <= max_gap:
            merged[-1][1]=end
        else:
            merged.append([start,end])
    return merged


def split_candidate_by_projection(
    frame,
    candidate,
    white_threshold=180,
    min_row_fraction=.005,
    min_run_height=5,
    search_up_ratio=.25,
    search_down_ratio=1.25,
    search_side_ratio=.50,
    max_lines=2,
    pixel_pad=2,
):
    if candidate.level != "HIGH":
        return [candidate]

    h_frame,w_frame=frame.shape[:2]
    box=np.asarray(candidate.bbox,dtype=np.float32)
    x1,y1,x2,y2=[float(v) for v in box]
    bw=max(1.0,x2-x1)
    bh=max(1.0,y2-y1)

    sx1=max(0,int(np.floor(x1-search_side_ratio*bh)))
    sx2=min(w_frame,int(np.ceil(x2+search_side_ratio*bh)))
    sy1=max(0,int(np.floor(y1-search_up_ratio*bh)))
    sy2=min(h_frame,int(np.ceil(y2+search_down_ratio*bh)))
    if sx2<=sx1 or sy2<=sy1:
        return [candidate]

    crop=frame[sy1:sy2,sx1:sx2]
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    mask=gray>=int(white_threshold)
    counts=mask.sum(axis=1)
    min_count=max(3,int(round(mask.shape[1]*float(min_row_fraction))))
    active=counts>=min_count

    runs=_merge_close_runs(_active_runs(active),max_gap=2)
    runs=[
        run for run in runs
        if run[1]-run[0]+1 >= int(min_run_height)
    ]
    if len(runs)<2:
        return [candidate]

    scored=[]
    for start,end in runs:
        band=mask[start:end+1]
        ys,xs=np.where(band)
        if len(xs)==0:
            continue
        score=float(counts[start:end+1].sum())
        scored.append((score,start,end,int(xs.min()),int(xs.max())))
    if len(scored)<2:
        return [candidate]

    # Keep the strongest lines but preserve top-to-bottom order.
    scored=sorted(scored,key=lambda x:x[0],reverse=True)[:int(max_lines)]
    scored=sorted(scored,key=lambda x:x[1])

    out=[]
    for _,start,end,xmin,xmax in scored:
        bx1=max(0,sx1+xmin-int(pixel_pad))
        by1=max(0,sy1+start-int(pixel_pad))
        bx2=min(w_frame,sx1+xmax+1+int(pixel_pad))
        by2=min(h_frame,sy1+end+1+int(pixel_pad))
        out.append(Candidate(
            np.array([bx1,by1,bx2,by2],np.float32),
            candidate.score,
            candidate.level,
            candidate.source,
        ))

    if len(out) != 2:
        return [candidate]

    top,bottom=out
    top_h=max(1.0,float(top.bbox[3]-top.bbox[1]))
    bottom_h=max(1.0,float(bottom.bbox[3]-bottom.bbox[1]))
    height_similarity=min(top_h,bottom_h)/max(top_h,bottom_h)
    top_cx=float((top.bbox[0]+top.bbox[2])*.5)
    bottom_cx=float((bottom.bbox[0]+bottom.bbox[2])*.5)
    center_dx_norm=abs(top_cx-bottom_cx)/bw
    vertical_gap=float(bottom.bbox[1]-top.bbox[3])

    if center_dx_norm > .15:
        return [candidate]
    if height_similarity < .75:
        return [candidate]
    if not -3.0 <= vertical_gap <= 15.0:
        return [candidate]

    return out


def split_frame_detections_by_projection(frame, detections):
    from .line_grouping import group_candidates_to_lines
    from .types import FrameDetections

    expanded=[]
    for candidate in list(detections.high)+list(detections.low):
        expanded.extend(split_candidate_by_projection(frame,candidate))

    grouped=group_candidates_to_lines(expanded)
    return FrameDetections(
        frame_index=detections.frame_index,
        timestamp=detections.timestamp,
        high=[c for c in grouped if c.level=="HIGH"],
        low=[c for c in grouped if c.level=="LOW"],
    )
