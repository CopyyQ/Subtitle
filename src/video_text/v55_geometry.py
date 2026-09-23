from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .text_enhancement import white_black_text_mask


@dataclass(slots=True)
class LineGeometry:
    bbox: np.ndarray
    confidence: float
    used_anchor: bool

    def __post_init__(self):
        self.bbox=np.asarray(self.bbox,dtype=np.float32)
        self.confidence=float(self.confidence)
        self.used_anchor=bool(self.used_anchor)


def line_overlap_area(a,b):
    a=np.asarray(a,dtype=np.float32)
    b=np.asarray(b,dtype=np.float32)
    x1=max(float(a[0]),float(b[0]))
    y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2]))
    y2=min(float(a[3]),float(b[3]))
    return max(0.0,x2-x1)*max(0.0,y2-y1)


def _row_activity(frame,x1,x2,y1,y2):
    h,w=frame.shape[:2]
    x1=max(0,min(w,int(np.floor(x1))))
    x2=max(0,min(w,int(np.ceil(x2))))
    y1=max(0,min(h,int(np.floor(y1))))
    y2=max(0,min(h,int(np.ceil(y2))))
    if x2<=x1 or y2<=y1:
        return np.zeros(max(0,y2-y1),dtype=np.float32)
    mask=white_black_text_mask(frame[y1:y2,x1:x2])
    return mask.sum(axis=1).astype(np.float32)


def estimate_separator_seam(frame,top,bottom,search_pad=4):
    top=np.asarray(top,dtype=np.float32)
    bottom=np.asarray(bottom,dtype=np.float32)
    overlap_x1=max(float(top[0]),float(bottom[0]))
    overlap_x2=min(float(top[2]),float(bottom[2]))
    lo=int(np.floor(min(float(top[3]),float(bottom[1]))-int(search_pad)))
    hi=int(np.ceil(max(float(top[3]),float(bottom[1]))+int(search_pad)))
    if hi<=lo:
        lo=int(np.floor(min(float(top[3]),float(bottom[3]))))
        hi=int(np.ceil(max(float(top[1]),float(bottom[1]))))
    if hi<=lo:
        return int(round((float(top[3])+float(bottom[1]))*.5))

    if overlap_x2<=overlap_x1:
        return int(round((float(top[3])+float(bottom[1]))*.5))

    activity=_row_activity(frame,overlap_x1,overlap_x2,lo,hi)
    if activity.size==0:
        return int(round((float(top[3])+float(bottom[1]))*.5))

    # Prefer the quietest row in the vertical conflict/gap region. Ties use
    # the geometric middle so the seam does not drift toward either line.
    minv=float(activity.min())
    ids=np.flatnonzero(activity==minv)
    mid=(len(activity)-1)*.5
    idx=int(ids[np.argmin(np.abs(ids-mid))]) if len(ids) else int(round(mid))
    return int(lo+idx)


def enforce_non_overlapping_lines(frame,boxes):
    ordered=[
        np.asarray(box,dtype=np.float32).copy()
        for box in boxes
    ]
    ordered.sort(key=lambda b:(float(b[1])+float(b[3]))*.5)

    for i in range(len(ordered)-1):
        top=ordered[i]
        bottom=ordered[i+1]
        if float(top[3]) <= float(bottom[1]):
            continue

        seam=estimate_separator_seam(frame,top,bottom)
        lower=int(np.ceil(float(top[1])+1.0))
        upper=int(np.floor(float(bottom[3])-1.0))
        seam=max(lower,min(upper,seam))
        top[3]=min(float(top[3]),float(seam))
        bottom[1]=max(float(bottom[1]),float(seam))

    return ordered
def _clip_bbox(box,width,height):
    b=np.asarray(box,dtype=np.float32).copy()
    b[0]=max(0.0,min(float(width),float(b[0])))
    b[2]=max(0.0,min(float(width),float(b[2])))
    b[1]=max(0.0,min(float(height),float(b[1])))
    b[3]=max(0.0,min(float(height),float(b[3])))
    return b


def refine_line_bbox(
    frame,
    anchor,
    slot,
    max_x_shift_ratio=.30,
    max_y_shift_ratio=.12,
    pad_px=2,
):
    anchor=np.asarray(anchor,dtype=np.float32)
    fh,fw=frame.shape[:2]
    h=max(1.0,float(slot.expected_height))
    x_margin=max(4.0,h*float(max_x_shift_ratio))
    y_margin=max(2.0,h*float(max_y_shift_ratio))
    cx1=max(0,int(np.floor(float(anchor[0])-x_margin)))
    cy1=max(0,int(np.floor(float(anchor[1])-y_margin)))
    cx2=min(fw,int(np.ceil(float(anchor[2])+x_margin)))
    cy2=min(fh,int(np.ceil(float(anchor[3])+y_margin)))
    if cx2<=cx1 or cy2<=cy1:
        return LineGeometry(anchor.copy(),0.0,True)
    crop=frame[cy1:cy2,cx1:cx2]
    mask=white_black_text_mask(crop)
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(mask,8)
    slot_cy=(float(slot.y1)+float(slot.y2))*.5
    keep=[]
    min_area=max(4.0,.0025*h*h)
    for i in range(1,n):
        x,y,wc,hc,area=stats[i]
        if float(area)<min_area or wc<2 or hc<2:
            continue
        abs_cy=cy1+float(centroids[i][1])
        if abs(abs_cy-slot_cy)>.48*h:
            continue
        bx1=float(cx1+x); by1=float(cy1+y)
        bx2=float(cx1+x+wc); by2=float(cy1+y+hc)
        if bx2 < float(anchor[0])-x_margin or bx1 > float(anchor[2])+x_margin:
            continue
        keep.append((i,bx1,by1,bx2,by2,float(area)))
    if not keep:
        return LineGeometry(anchor.copy(),0.0,True)

    # Keep the horizontally connected chain nearest the slot/anchor center.
    target_cx=float(slot.expected_x_center)
    keep.sort(key=lambda r:(r[1]+r[3])*.5)
    seed=min(range(len(keep)),key=lambda i:abs((keep[i][1]+keep[i][3])*.5-target_cx))
    selected=[keep[seed]]
    max_gap=.55*h
    left=seed-1
    current_left=keep[seed][1]
    while left>=0:
        gap=current_left-keep[left][3]
        if gap>max_gap:
            break
        selected.append(keep[left])
        current_left=min(current_left,keep[left][1])
        left-=1
    right=seed+1
    current_right=keep[seed][3]
    while right<len(keep):
        gap=keep[right][1]-current_right
        if gap>max_gap:
            break
        selected.append(keep[right])
        current_right=max(current_right,keep[right][3])
        right+=1

    x1=min(r[1] for r in selected)-float(pad_px)
    y1=min(r[2] for r in selected)-float(pad_px)
    x2=max(r[3] for r in selected)+float(pad_px)
    y2=max(r[4] for r in selected)+float(pad_px)
    box=_clip_bbox([x1,y1,x2,y2],fw,fh)

    # The temporal anchor is the trusted outer envelope for a subtitle line.
    # Per-frame evidence may tighten it, or move horizontally by only the
    # minimal outline pad, but scene-like white/dark pixels must not make the
    # line grow vertically beyond the locked slot.
    x_expand=max(2.0,float(pad_px))
    box[0]=max(box[0],float(anchor[0])-x_expand)
    box[2]=min(box[2],float(anchor[2])+x_expand)
    box[1]=max(box[1],float(anchor[1]))
    box[3]=min(box[3],float(anchor[3]))
    if box[2]<=box[0] or box[3]<=box[1]:
        return LineGeometry(anchor.copy(),0.0,True)

    evidence_area=sum(r[5] for r in selected)
    box_area=max(1.0,float((box[2]-box[0])*(box[3]-box[1])))
    confidence=min(1.0,evidence_area/max(1.0,.12*box_area))
    return LineGeometry(box,confidence,False)


def smooth_line_geometry(sequence,anchor,window=5,max_step_px=2.0):
    if not sequence:
        return []
    anchor=np.asarray(anchor,dtype=np.float32)
    arr=np.stack([
        np.asarray(item.bbox,dtype=np.float32)
        for item in sequence
    ],axis=0)
    radius=max(0,int(window)//2)
    targets=[]
    for i in range(len(arr)):
        lo=max(0,i-radius)
        hi=min(len(arr),i+radius+1)
        targets.append(np.median(arr[lo:hi],axis=0).astype(np.float32))
    out=[targets[0].copy()]
    max_step=float(max_step_px)
    for target in targets[1:]:
        prev=out[-1]
        delta=np.clip(target-prev,-max_step,max_step)
        cur=(prev+delta).astype(np.float32)
        if cur[2]<=cur[0] or cur[3]<=cur[1]:
            cur=prev.copy()
        out.append(cur)
    return out
