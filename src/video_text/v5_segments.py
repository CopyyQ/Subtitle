from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(slots=True)
class SubtitleSegment:
    start_frame: int
    end_frame: int
    track_ids: list[int]
    search_bbox: list[float]
    median_bbox: np.ndarray


def _track_summary(track):
    fs=track.sorted_frames()
    if not fs:
        return None
    boxes=np.stack([track.observations[f].bbox for f in fs],axis=0).astype(np.float32)
    med=np.median(boxes,axis=0).astype(np.float32)
    return {
        "track":track,
        "start":min(fs),
        "end":max(fs),
        "median":med,
        "content_boundary_before":bool(
            getattr(track,"content_boundary_before",False)
        ),
    }


def _iou(a,b):
    x1=max(float(a[0]),float(b[0]))
    y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2]))
    y2=min(float(a[3]),float(b[3]))
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    aa=max(0.0,float(a[2]-a[0]))*max(0.0,float(a[3]-a[1]))
    bb=max(0.0,float(b[2]-b[0]))*max(0.0,float(b[3]-b[1]))
    den=aa+bb-inter
    return inter/den if den>0 else 0.0


def _vertical_relation(a,b):
    ha=max(1.0,float(a[3]-a[1]))
    hb=max(1.0,float(b[3]-b[1]))
    ac=(float(a[1])+float(a[3]))*.5
    bc=(float(b[1])+float(b[3]))*.5
    return abs(ac-bc)/max(ha,hb)


def _same_temporal_block(a,b):
    overlap=min(a["end"],b["end"])-max(a["start"],b["start"])+1
    gap=max(a["start"],b["start"])-min(a["end"],b["end"])-1

    if overlap>0:
        # Overlapping tracks may be two subtitle rows. Permit a larger
        # baseline separation, but still require them to live in the same
        # subtitle neighborhood.
        return _vertical_relation(a["median"],b["median"]) <= 1.55

    if gap<=0:
        gap=0
    if gap>1:
        return False

    later=b if b["start"]>=a["start"] else a
    if later.get("content_boundary_before",False):
        return False

    # Adjacent fragments are merged only when their geometry is almost the
    # same. IoU alone is not enough: two different subtitle strings in the
    # same lane can still overlap heavily while both horizontal edges move by
    # roughly one character height.
    ah=max(1.0,float(a["median"][3]-a["median"][1]))
    bh=max(1.0,float(b["median"][3]-b["median"][1]))
    href=max(ah,bh)
    left_shift=abs(float(a["median"][0])-float(b["median"][0]))
    right_shift=abs(float(a["median"][2])-float(b["median"][2]))
    return (
        _iou(a["median"],b["median"]) >= .68
        and _vertical_relation(a["median"],b["median"]) <= .35
        and left_shift <= .65*href
        and right_shift <= .65*href
    )


def _component_summary(items):
    start=min(x["start"] for x in items)
    end=max(x["end"] for x in items)
    medians=np.stack([x["median"] for x in items],axis=0).astype(np.float32)
    # For a multi-line block, the union of median line boxes describes the
    # coarse subtitle block while keeping the temporal span common.
    union=np.array([
        float(np.min(medians[:,0])),
        float(np.min(medians[:,1])),
        float(np.max(medians[:,2])),
        float(np.max(medians[:,3])),
    ],np.float32)

    widths=medians[:,2]-medians[:,0]
    heights=medians[:,3]-medians[:,1]
    ref_h=float(np.median(heights))
    # A top-line-only FAST box must leave enough room for a missed second row.
    sx1=float(union[0]-.50*ref_h)
    sx2=float(union[2]+.50*ref_h)
    sy1=float(union[1]-.35*ref_h)
    sy2=float(union[3]+2.00*ref_h)

    return SubtitleSegment(
        start_frame=int(start),
        end_frame=int(end),
        track_ids=sorted(int(x["track"].track_id) for x in items),
        search_bbox=[sx1,sy1,sx2,sy2],
        median_bbox=union,
    )


def build_subtitle_segments(tracks):
    items=[x for x in (_track_summary(t) for t in tracks) if x is not None]
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
            if _same_temporal_block(items[i],items[j]):
                union(i,j)

    groups={}
    for i,item in enumerate(items):
        groups.setdefault(find(i),[]).append(item)

    segments=[_component_summary(group) for group in groups.values()]
    segments.sort(key=lambda s:(s.start_frame,s.end_frame))
    return segments
