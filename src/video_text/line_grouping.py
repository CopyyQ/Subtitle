from __future__ import annotations
import numpy as np
from .types import Candidate

def _vertical_overlap(a,b):
    inter=max(0.0,min(a[3],b[3])-max(a[1],b[1]))
    h1=max(1.0,float(a[3]-a[1]))
    h2=max(1.0,float(b[3]-b[1]))
    return inter/min(h1,h2)

def _compatible(a,b,max_gap_heights=4.0):
    ha=max(1.0,float(a.bbox[3]-a.bbox[1]))
    hb=max(1.0,float(b.bbox[3]-b.bbox[1]))
    big=max(ha,hb)
    small=min(ha,hb)
    ratio=small/big
    cya=float((a.bbox[1]+a.bbox[3])*.5)
    cyb=float((b.bbox[1]+b.bbox[3])*.5)

    normal_vertical=(
        _vertical_overlap(a.bbox,b.bbox)>=.35
        or abs(cya-cyb)<=.45*big
    )
    thin_low=(a.level=="LOW" or b.level=="LOW") and ratio>=.10
    if ratio>=.50:
        vertical=normal_vertical
    elif thin_low:
        large=a if ha>=hb else b
        small_cy=cyb if ha>=hb else cya
        margin=.20*big
        vertical=(
            float(large.bbox[1])-margin
            <= small_cy
            <= float(large.bbox[3])+margin
        )
    else:
        return False
    if not vertical:
        return False

    gap=max(
        0.0,
        max(float(a.bbox[0]),float(b.bbox[0]))
        - min(float(a.bbox[2]),float(b.bbox[2])),
    )
    allowed_gap_heights=1.5 if thin_low and ratio<.50 else float(max_gap_heights)
    return gap<=allowed_gap_heights*big

def group_candidates_to_lines(candidates,max_gap_heights=4.0):
    items=sorted(
        list(candidates),
        key=lambda c:(
            0 if c.level=="HIGH" else 1,
            (c.bbox[1]+c.bbox[3])*.5,
            c.bbox[0],
        ),
    )
    groups=[]
    for cand in items:
        placed=False
        for group in groups:
            merged_bbox=np.array([
                min(x.bbox[0] for x in group),
                min(x.bbox[1] for x in group),
                max(x.bbox[2] for x in group),
                max(x.bbox[3] for x in group),
            ],np.float32)
            proxy=Candidate(
                merged_bbox,
                max(x.score for x in group),
                "HIGH" if any(x.level=="HIGH" for x in group) else "LOW",
            )
            if _compatible(proxy,cand,max_gap_heights=max_gap_heights):
                group.append(cand)
                placed=True
                break
        if not placed:
            groups.append([cand])

    out=[]
    for group in groups:
        bbox=np.array([
            min(x.bbox[0] for x in group),
            min(x.bbox[1] for x in group),
            max(x.bbox[2] for x in group),
            max(x.bbox[3] for x in group),
        ],np.float32)
        level="HIGH" if any(x.level=="HIGH" for x in group) else "LOW"
        score=max(x.score for x in group)
        source="+".join(sorted(set(x.source for x in group)))
        out.append(Candidate(bbox,score,level,source))
    return sorted(out,key=lambda c:(c.bbox[1],c.bbox[0]))

def _box_area(box):
    return max(0.0,float(box[2]-box[0]))*max(0.0,float(box[3]-box[1]))

def _intersection_area(a,b):
    x1=max(float(a[0]),float(b[0]))
    y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2]))
    y2=min(float(a[3]),float(b[3]))
    return max(0.0,x2-x1)*max(0.0,y2-y1)

def suppress_nested_candidates(candidates,containment_threshold=.92,max_area_ratio=.35):
    items=list(candidates)
    if len(items)<2:
        return items
    order=sorted(
        range(len(items)),
        key=lambda i:(_box_area(items[i].bbox),items[i].score),
        reverse=True,
    )
    keep=[]
    for idx in order:
        cand=items[idx]
        ca=_box_area(cand.bbox)
        nested=False
        for kept_idx in keep:
            outer=items[kept_idx]
            oa=_box_area(outer.bbox)
            if ca<=0 or oa<=0 or ca>oa*max_area_ratio:
                continue
            contained=_intersection_area(cand.bbox,outer.bbox)/ca
            if contained>=containment_threshold:
                nested=True
                break
        if not nested:
            keep.append(idx)
    keep_set=set(keep)
    return [c for i,c in enumerate(items) if i in keep_set]
