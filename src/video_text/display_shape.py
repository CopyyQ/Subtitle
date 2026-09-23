from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import cv2


def _normalize_rect(rect):
    x1,y1,x2,y2=[float(v) for v in rect]
    if x2<x1:
        x1,x2=x2,x1
    if y2<y1:
        y1,y2=y2,y1
    if x2<=x1 or y2<=y1:
        return None
    return [x1,y1,x2,y2]


def _signed_area(poly):
    if len(poly)<3:
        return 0.0
    s=0.0
    for a,b in zip(poly,poly[1:]+poly[:1]):
        s+=a[0]*b[1]-b[0]*a[1]
    return .5*s


def _simplify_collinear(poly):
    pts=list(poly)
    if len(pts)<=4:
        return pts
    changed=True
    while changed and len(pts)>4:
        changed=False
        out=[]
        n=len(pts)
        for i,p in enumerate(pts):
            prev=pts[(i-1)%n]
            nxt=pts[(i+1)%n]
            if (
                (prev[0]==p[0]==nxt[0])
                or (prev[1]==p[1]==nxt[1])
            ):
                changed=True
                continue
            out.append(p)
        pts=out
    return pts


def rectangle_union_polygons(rectangles: Iterable[Iterable[float]]):
    rects=[r for r in (_normalize_rect(x) for x in rectangles) if r is not None]
    if not rects:
        return []

    xs=sorted({v for r in rects for v in (r[0],r[2])})
    ys=sorted({v for r in rects for v in (r[1],r[3])})
    nx=max(0,len(xs)-1)
    ny=max(0,len(ys)-1)
    occupied=np.zeros((ny,nx),dtype=bool)

    for j in range(ny):
        cy=(ys[j]+ys[j+1])*.5
        for i in range(nx):
            cx=(xs[i]+xs[i+1])*.5
            occupied[j,i]=any(
                r[0] <= cx <= r[2] and r[1] <= cy <= r[3]
                for r in rects
            )

    edges=[]
    for j in range(ny):
        for i in range(nx):
            if not occupied[j,i]:
                continue
            x1,x2=xs[i],xs[i+1]
            y1,y2=ys[j],ys[j+1]
            if j==0 or not occupied[j-1,i]:
                edges.append(((x1,y1),(x2,y1)))
            if i==nx-1 or not occupied[j,i+1]:
                edges.append(((x2,y1),(x2,y2)))
            if j==ny-1 or not occupied[j+1,i]:
                edges.append(((x2,y2),(x1,y2)))
            if i==0 or not occupied[j,i-1]:
                edges.append(((x1,y2),(x1,y1)))

    outgoing=defaultdict(list)
    for a,b in edges:
        outgoing[a].append(b)

    unused=set(edges)
    polygons=[]
    while unused:
        start_edge=next(iter(unused))
        start=start_edge[0]
        current=start
        poly=[start]
        while True:
            candidates=[b for b in outgoing[current] if (current,b) in unused]
            if not candidates:
                break
            nxt=candidates[0]
            unused.remove((current,nxt))
            current=nxt
            if current==start:
                break
            poly.append(current)
        if current==start and len(poly)>=4:
            poly=_simplify_collinear(poly)
            polygons.append(poly)

    polygons.sort(key=lambda p:abs(_signed_area(p)),reverse=True)
    return polygons


def _horizontal_overlap_ratio(a,b):
    overlap=max(0.0,min(a[2],b[2])-max(a[0],b[0]))
    wa=max(1.0,a[2]-a[0])
    wb=max(1.0,b[2]-b[0])
    return overlap/min(wa,wb)


def _vertical_gap(a,b):
    if a[3] < b[1]:
        return b[1]-a[3]
    if b[3] < a[1]:
        return a[1]-b[3]
    return 0.0


def _same_subtitle_block(a,b,join_gap_ratio=.30,min_x_overlap_ratio=.30):
    ha=max(1.0,a[3]-a[1])
    hb=max(1.0,b[3]-b[1])
    return (
        _vertical_gap(a,b) <= join_gap_ratio*max(ha,hb)
        and _horizontal_overlap_ratio(a,b) >= min_x_overlap_ratio
    )


def _connect_close_lines(rectangles):
    rects=[list(r) for r in rectangles]
    order=sorted(range(len(rects)),key=lambda i:(rects[i][1]+rects[i][3])*.5)
    for upper_i,lower_i in zip(order,order[1:]):
        upper=rects[upper_i]
        lower=rects[lower_i]
        gap=lower[1]-upper[3]
        if gap<=0:
            continue
        if _horizontal_overlap_ratio(upper,lower)<=0:
            continue
        seam=(upper[3]+lower[1])*.5
        upper[3]=seam
        lower[1]=seam
    return rects


def build_display_polygons(
    records,
    join_gap_ratio=.30,
    min_x_overlap_ratio=.30,
):
    items=[]
    for r in records:
        rect=_normalize_rect(r["bbox"])
        if rect is None:
            continue
        items.append({"track_id":int(r["track_id"]),"bbox":rect})

    if not items:
        return []

    parent=list(range(len(items)))

    def find(x):
        while parent[x]!=x:
            parent[x]=parent[parent[x]]
            x=parent[x]
        return x

    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb:
            parent[rb]=ra

    for i in range(len(items)):
        for j in range(i+1,len(items)):
            if _same_subtitle_block(
                items[i]["bbox"],items[j]["bbox"],
                join_gap_ratio=join_gap_ratio,
                min_x_overlap_ratio=min_x_overlap_ratio,
            ):
                union(i,j)

    groups=defaultdict(list)
    for i,item in enumerate(items):
        groups[find(i)].append(item)

    shapes=[]
    for group in groups.values():
        rects=[x["bbox"] for x in group]
        if len(rects)>1:
            rects=_connect_close_lines(rects)
        polygons=rectangle_union_polygons(rects)
        for poly in polygons:
            shapes.append({
                "track_ids":sorted(x["track_id"] for x in group),
                "polygon":[
                    [int(round(x)),int(round(y))]
                    for x,y in poly
                ],
            })

    shapes.sort(key=lambda s:(min(p[1] for p in s["polygon"]),min(p[0] for p in s["polygon"])))
    return shapes


def draw_display_polygons(
    frame,
    records,
    color=(0,255,0),
    thickness=2,
    line_type=cv2.LINE_AA,
):
    shapes=build_display_polygons(records)
    for shape in shapes:
        pts=np.asarray(shape["polygon"],dtype=np.int32).reshape(-1,1,2)
        if len(pts)>=3:
            cv2.polylines(frame,[pts],True,color,int(thickness),line_type)
    return shapes


def build_display_shape_records(records, fps):
    by_frame=defaultdict(list)
    for r in records:
        by_frame[int(r["frame"])].append(r)
    rows=[]
    for frame_index in sorted(by_frame):
        for shape in build_display_polygons(by_frame[frame_index]):
            rows.append({
                "frame":frame_index,
                "timestamp":frame_index/float(fps),
                "track_ids":shape["track_ids"],
                "polygon":shape["polygon"],
            })
    return rows


def draw_line_rectangles(
    frame,
    records,
    color=(0,255,0),
    thickness=2,
    line_type=cv2.LINE_AA,
):
    shapes=[]
    for record in records:
        rect=_normalize_rect(record.get("bbox"))
        if rect is None:
            continue
        x1,y1,x2,y2=[int(round(v)) for v in rect]
        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            color,
            int(thickness),
            line_type,
        )
        shapes.append({
            "subtitle_id":int(record.get("subtitle_id",record.get("track_id",0))),
            "line_id":int(record.get("line_id",0)),
            "track_id":int(record.get("track_id",0)),
            "polygon":[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
        })
    return shapes


def build_line_shape_records(records,fps):
    by_frame=defaultdict(list)
    for r in records:
        by_frame[int(r["frame"])].append(r)
    rows=[]
    for frame_index in sorted(by_frame):
        for r in by_frame[frame_index]:
            rect=_normalize_rect(r.get("bbox"))
            if rect is None:
                continue
            x1,y1,x2,y2=[int(round(v)) for v in rect]
            rows.append({
                "frame":frame_index,
                "timestamp":frame_index/float(fps),
                "subtitle_id":int(r.get("subtitle_id",r.get("track_id",0))),
                "line_id":int(r.get("line_id",0)),
                "track_id":int(r.get("track_id",0)),
                "track_ids":[int(r.get("track_id",0))],
                "polygon":[[x1,y1],[x2,y1],[x2,y2],[x1,y2]],
            })
    return rows
