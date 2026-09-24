from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from .types import Candidate, FrameDetections, SubtitleTrack, TrackObservation
from .line_grouping import suppress_nested_candidates

@dataclass(slots=True)
class AssociationConfig:
    iou_gate: float=.20
    center_gate: float=.04
    min_width_ratio: float=.65
    max_width_ratio: float=1.55
    min_height_ratio: float=.65
    max_height_ratio: float=1.55
    max_frame_gap: int=3
    backfill_frames: int=2

def iou(a,b):
    a=np.asarray(a,np.float32)
    b=np.asarray(b,np.float32)
    x1=max(float(a[0]),float(b[0]))
    y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2]))
    y2=min(float(a[3]),float(b[3]))
    inter=max(0.,x2-x1)*max(0.,y2-y1)
    aa=max(0.,float(a[2]-a[0]))*max(0.,float(a[3]-a[1]))
    bb=max(0.,float(b[2]-b[0]))*max(0.,float(b[3]-b[1]))
    return inter/max(aa+bb-inter,1e-6)

def _ratio(v1,v2):
    return float(v1)/max(float(v2),1e-6)

def compatible(a,b,cfg,frame_diag=1468.0):
    a=np.asarray(a,np.float32)
    b=np.asarray(b,np.float32)
    wa=max(1.,float(a[2]-a[0]))
    wb=max(1.,float(b[2]-b[0]))
    ha=max(1.,float(a[3]-a[1]))
    hb=max(1.,float(b[3]-b[1]))
    wr=_ratio(wb,wa)
    hr=_ratio(hb,ha)
    if not (cfg.min_width_ratio<=wr<=cfg.max_width_ratio and
            cfg.min_height_ratio<=hr<=cfg.max_height_ratio):
        return False
    ac=(a[:2]+a[2:])*.5
    bc=(b[:2]+b[2:])*.5

    # A subtitle line must stay on the same vertical baseline. Without this
    # guard a top-line track can jump into a lower subtitle line because the
    # full-frame normalized center distance is still small.
    vertical_overlap=max(
        0.0,
        min(float(a[3]),float(b[3]))-max(float(a[1]),float(b[1])),
    )/max(1.0,min(ha,hb))
    center_y_delta=abs(float(ac[1]-bc[1]))
    if vertical_overlap < .20 and center_y_delta > .55*max(ha,hb):
        return False

    cd=float(np.linalg.norm(ac-bc))/max(frame_diag,1.)
    return iou(a,b)>=cfg.iou_gate or cd<=cfg.center_gate

def _cost(a,b,cfg):
    ac=(a[:2]+a[2:])*.5
    bc=(b[:2]+b[2:])*.5
    scale=max(float(np.linalg.norm(a[2:]-a[:2])),1.)
    cd=float(np.linalg.norm(ac-bc))/scale
    return (1.-iou(a,b))+.25*cd

def _last_obs(track):
    return track.observations[max(track.observations)]

def build_provisional_tracks(frames, config=None):
    cfg=config or AssociationConfig()
    tracks=[]
    next_id=1
    pending_low=[]
    for frame in sorted(frames,key=lambda x:x.frame_index):
        fi=frame.frame_index
        pending_low=[x for x in pending_low if fi-x[0] <= cfg.backfill_frames]
        active=[
            t for t in tracks
            if t.confirmed and t.observations and fi-max(t.observations)<=cfg.max_frame_gap
        ]
        candidates=suppress_nested_candidates(list(frame.high)+list(frame.low))
        matched_c=set()
        if active and candidates:
            cost=np.full((len(active),len(candidates)),1e6,np.float32)
            for i,t in enumerate(active):
                prev=_last_obs(t).bbox
                for j,c in enumerate(candidates):
                    if compatible(prev,c.bbox,cfg):
                        cost[i,j]=_cost(prev,c.bbox,cfg)
            rows,cols=linear_sum_assignment(cost)
            for i,j in zip(rows,cols):
                if cost[i,j]>=1e5:
                    continue
                t=active[i]
                c=candidates[j]
                t.observations[fi]=TrackObservation(fi,c.bbox,c.score,c.level)
                matched_c.add(j)
        for j,c in enumerate(candidates):
            if j in matched_c:
                continue
            if c.level=="LOW":
                pending_low.append((fi,c))
                continue
            t=SubtitleTrack(next_id,confirmed=True)
            next_id+=1
            t.observations[fi]=TrackObservation(fi,c.bbox,c.score,"HIGH")
            for pfi,pc in sorted(pending_low,key=lambda x:x[0],reverse=True):
                if 0 < fi-pfi <= cfg.backfill_frames and compatible(pc.bbox,c.bbox,cfg):
                    t.observations[pfi]=TrackObservation(pfi,pc.bbox,pc.score,"LOW")
            if t.observations:
                earliest=min(t.observations)
                pending_low=[
                    x for x in pending_low
                    if x[0]!=earliest or not np.allclose(x[1].bbox,t.observations[earliest].bbox)
                ]
            tracks.append(t)
    return tracks
