from __future__ import annotations

import cv2
import numpy as np

from .types import Candidate


def _column_runs(active):
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
        if start-merged[-1][1]-1 <= int(max_gap):
            merged[-1][1]=end
        else:
            merged.append([start,end])
    return merged


def recover_horizontal_extent(
    frame,
    candidate,
    white_threshold=180,
    vertical_pad_ratio=.15,
    search_left_ratio=2.0,
    search_right_ratio=1.0,
    min_col_pixels=2,
    min_run_width_ratio=.08,
    max_char_gap_ratio=.35,
    max_extension_ratio=1.0,
    thin_extension_ratio=1.5,
    thin_run_height_ratio=.25,
    max_vertical_span_ratio=1.05,
    max_center_y_ratio=.35,
):
    """Expand a confirmed subtitle line to missed white glyphs on the same baseline.

    Recovery changes only x1/x2. It is conservative: external white runs must
    look like text on the same baseline and total extension is capped to about
    one line height per side.
    """
    if candidate.level != "HIGH":
        return candidate

    box=np.asarray(candidate.bbox,dtype=np.float32)
    x1,y1,x2,y2=[float(v) for v in box]
    line_h=max(1.0,y2-y1)
    line_cy=(y1+y2)*.5
    frame_h,frame_w=frame.shape[:2]

    sx1=max(0,int(np.floor(x1-search_left_ratio*line_h)))
    sx2=min(frame_w,int(np.ceil(x2+search_right_ratio*line_h)))
    sy1=max(0,int(np.floor(y1-vertical_pad_ratio*line_h)))
    sy2=min(frame_h,int(np.ceil(y2+vertical_pad_ratio*line_h)))
    if sx2<=sx1 or sy2<=sy1:
        return candidate

    crop=frame[sy1:sy2,sx1:sx2]
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    mask=gray>=int(white_threshold)
    counts=mask.sum(axis=0)
    active=counts>=int(max(1,min_col_pixels))
    raw_runs=_merge_close_runs(_column_runs(active),max_gap=2)

    min_width=max(3,int(round(line_h*float(min_run_width_ratio))))
    runs=[]
    for start,end in raw_runs:
        if end-start+1 < min_width:
            continue
        band=mask[:,start:end+1]
        yy,xx=np.where(band)
        if len(yy)==0:
            continue
        runs.append({
            "start":float(sx1+start),
            "end":float(sx1+end),
            "y_mean":float(sy1+yy.mean()),
            "y_span":float(yy.max()-yy.min()+1),
        })
    if not runs:
        return candidate

    seed=[
        i for i,run in enumerate(runs)
        if run["end"] >= x1 and run["start"] <= x2
    ]
    if not seed:
        return candidate

    max_gap=float(max_char_gap_ratio)*line_h
    max_extension=float(max_extension_ratio)*line_h
    thin_extension=float(thin_extension_ratio)*line_h

    def plausible_external(run):
        if run["y_span"] > float(max_vertical_span_ratio)*line_h:
            return False
        if abs(run["y_mean"]-line_cy) > float(max_center_y_ratio)*line_h:
            return False
        return True

    def extension_limit(run):
        if run["y_span"] <= float(thin_run_height_ratio)*line_h:
            return thin_extension
        return max_extension

    left=min(seed)
    right=max(seed)

    # A seed run may straddle the detector edge. Use its outside portion only
    # when it itself looks like same-baseline text and stays within the cap.
    recovered_x1=x1
    seed_left=runs[left]
    if (
        seed_left["start"] < x1
        and x1-seed_left["start"] <= extension_limit(seed_left)
        and plausible_external(seed_left)
    ):
        recovered_x1=seed_left["start"]

    recovered_x2=x2
    seed_right=runs[right]
    if (
        seed_right["end"]+1 > x2
        and seed_right["end"]+1-x2 <= extension_limit(seed_right)
        and plausible_external(seed_right)
    ):
        recovered_x2=seed_right["end"]+1

    # Chain neighboring glyph-like runs outward. Stop as soon as geometry no
    # longer resembles the same subtitle baseline or total extension is too large.
    while left>0:
        prev=runs[left-1]
        cur=runs[left]
        gap=float(cur["start"]-prev["end"]-1)
        total_extension=x1-prev["start"]
        if gap>max_gap or total_extension>extension_limit(prev) or not plausible_external(prev):
            break
        left-=1
        recovered_x1=min(recovered_x1,prev["start"])

    while right+1<len(runs):
        cur=runs[right]
        nxt=runs[right+1]
        gap=float(nxt["start"]-cur["end"]-1)
        total_extension=nxt["end"]+1-x2
        if gap>max_gap or total_extension>extension_limit(nxt) or not plausible_external(nxt):
            break
        right+=1
        recovered_x2=max(recovered_x2,nxt["end"]+1)

    if recovered_x1==x1 and recovered_x2==x2:
        return candidate

    return Candidate(
        np.array([recovered_x1,y1,recovered_x2,y2],np.float32),
        candidate.score,
        candidate.level,
        candidate.source,
    )


def recover_frame_horizontal_extents(frame, detections):
    from .types import FrameDetections

    high=[recover_horizontal_extent(frame,c) for c in detections.high]
    low=list(detections.low)
    return FrameDetections(
        frame_index=detections.frame_index,
        timestamp=detections.timestamp,
        high=high,
        low=low,
    )


def recover_tracks_horizontal_extents(video_path, tracks):
    import copy

    out=copy.deepcopy(tracks)
    by_frame={}
    max_frame=-1
    for track in out:
        for frame_index,obs in track.observations.items():
            if obs.reconstructed or obs.level != "HIGH":
                continue
            by_frame.setdefault(int(frame_index),[]).append(obs)
            max_frame=max(max_frame,int(frame_index))

    if max_frame < 0:
        return out,0

    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode video for horizontal recovery: {video_path}")

    expanded=0
    for frame_index in range(max_frame+1):
        ok,frame=cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(
                f"cannot decode frame {frame_index} for horizontal recovery"
            )
        for obs in by_frame.get(frame_index,[]):
            candidate=Candidate(
                obs.bbox,
                obs.score if obs.score is not None else 1.0,
                "HIGH",
                "track",
            )
            recovered=recover_horizontal_extent(frame,candidate)
            if not np.allclose(recovered.bbox,obs.bbox):
                obs.bbox=recovered.bbox.copy()
                expanded+=1

    cap.release()
    return out,expanded
