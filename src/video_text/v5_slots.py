from __future__ import annotations

import math
import cv2
import numpy as np

from .v5_weak_birth import SlotPrior
from .text_enhancement import white_black_text_mask


def _runs(active):
    out=[]
    start=None
    vals=active.tolist()+[False]
    for i,v in enumerate(vals):
        if v and start is None:
            start=i
        elif not v and start is not None:
            out.append([start,i-1])
            start=None
    return out


def _merge_runs(runs,max_gap=2):
    if not runs:
        return []
    merged=[runs[0][:]]
    for start,end in runs[1:]:
        if start-merged[-1][1]-1 <= int(max_gap):
            merged[-1][1]=end
        else:
            merged.append([start,end])
    return merged


def _text_core_mask(frame,white_threshold=180,dark_threshold=90,outline_radius=3):
    # V5.1+ enhancement: local contrast normalization + outline-aware mask.
    # Parameters are retained for API compatibility but the enhanced mask
    # intentionally lowers the effective white threshold in a controlled way.
    return white_black_text_mask(
        frame,
        bright_threshold=min(160,int(white_threshold)),
        dark_threshold=max(95,int(dark_threshold)),
        outline_radius=int(outline_radius),
    )


def bootstrap_line_slots(
    frames,
    search_bbox,
    support_ratio=.43,
    min_run_height=5,
    outline_pad=3,
    max_slots=2,
):
    """Infer stable vertical subtitle slots from several bootstrap frames.

    Only pixels that look like a white text core adjacent to dark outline are
    allowed to vote. A row must repeat in several bootstrap frames before it
    can create a slot, so one-frame background noise cannot become a line.
    """
    frames=list(frames)
    if not frames:
        return []

    fh,fw=frames[0].shape[:2]
    x1,y1,x2,y2=[int(round(v)) for v in search_bbox]
    x1=max(0,min(fw,x1)); x2=max(0,min(fw,x2))
    y1=max(0,min(fh,y1)); y2=max(0,min(fh,y2))
    if x2<=x1 or y2<=y1:
        return []

    masks=[]
    for frame in frames:
        crop=frame[y1:y2,x1:x2]
        masks.append(_text_core_mask(crop))
    stack=np.stack(masks,axis=0)
    need=max(2,int(math.ceil(len(masks)*float(support_ratio))))
    vote=(stack.sum(axis=0)>=need).astype(np.uint8)

    row_counts=vote.sum(axis=1)
    min_row_pixels=max(3,int(round(vote.shape[1]*.005)))
    active=row_counts>=min_row_pixels
    runs=_merge_runs(_runs(active),max_gap=2)
    runs=[
        r for r in runs
        if r[1]-r[0]+1 >= int(min_run_height)
    ]
    if not runs:
        return []

    scored=[]
    for start,end in runs:
        band=vote[start:end+1]
        ys,xs=np.where(band>0)
        if len(xs)==0:
            continue
        score=float(band.sum())
        abs_x1=x1+int(xs.min())
        abs_x2=x1+int(xs.max())+1
        abs_y1=y1+int(start)
        abs_y2=y1+int(end)+1
        scored.append({
            "score":score,
            "x1":abs_x1,"x2":abs_x2,
            "y1":abs_y1,"y2":abs_y2,
        })

    if not scored:
        return []

    # Keep strongest rows, but preserve their vertical order.
    scored=sorted(scored,key=lambda r:r["score"],reverse=True)[:int(max_slots)]
    scored=sorted(scored,key=lambda r:r["y1"])

    # Real two-line subtitles use the same font size on both rows. A much
    # shorter persistent run below the line is usually clothing/background
    # detail that survives temporal voting; it must not become a second slot.
    if len(scored)==2:
        heights=[max(1,r["y2"]-r["y1"]) for r in scored]
        similarity=min(heights)/max(heights)
        if similarity < .65:
            scored=[scored[int(np.argmax(heights))]]

    slots=[]
    for row in scored:
        sy1=max(0,row["y1"]-int(outline_pad))
        sy2=min(fh,row["y2"]+int(outline_pad))
        height=max(1,sy2-sy1)
        slots.append(SlotPrior(
            y1=sy1,
            y2=sy2,
            expected_x_center=(row["x1"]+row["x2"])*.5,
            expected_height=float(height),
        ))
    return slots
