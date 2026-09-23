from __future__ import annotations
from dataclasses import dataclass
import copy
import numpy as np
from .types import SubtitleTrack, TrackObservation

@dataclass(frozen=True,slots=True)
class TemporalEvent:
    event_type: str
    track_id: int
    start_frame: int
    end_frame: int

def _box_iou(a,b):
    x1=max(float(a[0]),float(b[0]))
    y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2]))
    y2=min(float(a[3]),float(b[3]))
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    aa=max(0.0,float(a[2]-a[0]))*max(0.0,float(a[3]-a[1]))
    bb=max(0.0,float(b[2]-b[0]))*max(0.0,float(b[3]-b[1]))
    return inter/max(aa+bb-inter,1e-6)

def _size_similarity(a,b):
    wa=max(1.0,float(a[2]-a[0]))
    wb=max(1.0,float(b[2]-b[0]))
    ha=max(1.0,float(a[3]-a[1]))
    hb=max(1.0,float(b[3]-b[1]))
    return min(wa,wb)/max(wa,wb), min(ha,hb)/max(ha,hb)

def split_tracks_on_geometry(
    tracks,
    max_internal_gap=2,
    gap_iou_min=.82,
    gap_size_similarity_min=.84,
    adjacent_iou_min=.78,
    adjacent_size_similarity_min=.85,
):
    if not tracks:
        return []
    next_id=max(t.track_id for t in tracks)+1
    output=[]
    for track in tracks:
        frames=track.sorted_frames()
        if not frames:
            continue
        groups=[[frames[0]]]
        for left,right in zip(frames,frames[1:]):
            a=track.observations[left].bbox
            b=track.observations[right].bbox
            ov=_box_iou(a,b)
            ws,_=_size_similarity(a,b)
            missing=right-left-1
            if missing>max_internal_gap:
                split=True
            elif missing>=1:
                split=ov<gap_iou_min or ws<gap_size_similarity_min
            else:
                split=ov<adjacent_iou_min and ws<adjacent_size_similarity_min
            if split:
                groups.append([right])
            else:
                groups[-1].append(right)
        for gi,group in enumerate(groups):
            part=copy.deepcopy(track)
            part.track_id=track.track_id if gi==0 else next_id
            if gi>0:
                next_id+=1
            part.observations={f:copy.deepcopy(track.observations[f]) for f in group}
            output.append(part)
    return output

def reconstruct_track(track, max_gap=2, total_frames=None):
    out=copy.deepcopy(track)
    events=[]
    frames=sorted(out.observations)
    for left,right in zip(frames,frames[1:]):
        gap=right-left-1
        if not 1<=gap<=max_gap:
            continue
        a=out.observations[left].bbox
        b=out.observations[right].bbox
        for k in range(1,gap+1):
            q=k/(gap+1.0)
            bbox=(1.-q)*a+q*b
            out.observations[left+k]=TrackObservation(
                left+k,bbox,None,"reconstructed",True
            )
        events.append(TemporalEvent("internal_miss_recovered",out.track_id,left+1,right-1))
    frames=sorted(out.observations)
    if frames and out.observations[frames[0]].level=="LOW":
        if any(out.observations[f].level=="HIGH" for f in frames[1:3]):
            first_high=next(f for f in frames if out.observations[f].level=="HIGH")
            events.append(TemporalEvent("track_start_backfill",out.track_id,frames[0],first_high-1))
    if total_frames is not None and frames and frames[-1] < total_frames-1:
        events.append(TemporalEvent("track_end",out.track_id,frames[-1],frames[-1]))
    return out,events

def reconstruct_tracks(tracks,max_gap=2,total_frames=None):
    out=[]
    events=[]
    for track in tracks:
        t,e=reconstruct_track(track,max_gap=max_gap,total_frames=total_frames)
        out.append(t)
        events.extend(e)
    return out,events


def _intersection_area(a,b):
    x1=max(float(a[0]),float(b[0]))
    y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2]))
    y2=min(float(a[3]),float(b[3]))
    return max(0.0,x2-x1)*max(0.0,y2-y1)


def _box_area(box):
    return (
        max(0.0,float(box[2]-box[0]))
        * max(0.0,float(box[3]-box[1]))
    )


def _vertical_overlap_ratio(a,b):
    overlap=max(
        0.0,
        min(float(a[3]),float(b[3]))
        - max(float(a[1]),float(b[1])),
    )
    ha=max(1.0,float(a[3]-a[1]))
    hb=max(1.0,float(b[3]-b[1]))
    return overlap/min(ha,hb)


def _contiguous_segments(frames):
    if not frames:
        return []
    ordered=sorted(frames)
    segments=[]
    start=prev=ordered[0]
    for frame in ordered[1:]:
        if frame==prev+1:
            prev=frame
            continue
        segments.append((start,prev))
        start=prev=frame
    segments.append((start,prev))
    return segments


def suppress_reconstructed_overlaps(
    tracks,
    events,
    coverage_threshold=.75,
    vertical_overlap_threshold=.80,
):
    """Remove bridged boxes when a real observation already occupies that line.

    A reconstructed observation represents a detector miss only when the frame
    has no actual observation covering the same subtitle-line region. This
    prevents track-fragment transitions from creating duplicate/ghost boxes.
    """
    out=copy.deepcopy(tracks)
    actual_by_frame={}
    for track in out:
        for frame_index,obs in track.observations.items():
            if obs.reconstructed:
                continue
            actual_by_frame.setdefault(int(frame_index),[]).append(
                (track.track_id,obs)
            )

    suppressed=0
    for track in out:
        remove=[]
        for frame_index,obs in track.observations.items():
            if not obs.reconstructed:
                continue
            recon_area=_box_area(obs.bbox)
            if recon_area<=0:
                continue
            for other_track_id,actual in actual_by_frame.get(int(frame_index),[]):
                if other_track_id==track.track_id:
                    continue
                if _vertical_overlap_ratio(obs.bbox,actual.bbox) < vertical_overlap_threshold:
                    continue
                coverage=_intersection_area(obs.bbox,actual.bbox)/recon_area
                if coverage>=coverage_threshold:
                    remove.append(frame_index)
                    break
        for frame_index in remove:
            del track.observations[frame_index]
            suppressed+=1

    by_id={track.track_id:track for track in out}
    filtered_events=[]
    for event in events:
        if event.event_type!="internal_miss_recovered":
            filtered_events.append(event)
            continue
        track=by_id.get(event.track_id)
        surviving=[]
        if track is not None:
            for frame_index in range(event.start_frame,event.end_frame+1):
                obs=track.observations.get(frame_index)
                if obs is not None and obs.reconstructed:
                    surviving.append(frame_index)
        for start,end in _contiguous_segments(surviving):
            filtered_events.append(
                TemporalEvent(
                    "internal_miss_recovered",
                    event.track_id,
                    start,
                    end,
                )
            )

    return out,filtered_events,suppressed
