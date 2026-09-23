from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from .types import SubtitleTrack
from .v5_slots import _text_core_mask


def _mask_similarity(a,b):
    kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    ad=cv2.dilate(a,kernel)
    bd=cv2.dilate(b,kernel)
    inter=int(np.logical_and(ad>0,bd>0).sum())
    union=int(np.logical_or(ad>0,bd>0).sum())
    return inter/max(1,union)


def _track_crop(track,frame_width,frame_height):
    fs=track.sorted_frames()
    boxes=np.stack([track.observations[f].bbox for f in fs],axis=0)
    med=np.median(boxes,axis=0).astype(np.float32)
    h=max(1.0,float(med[3]-med[1]))
    x1=max(0,int(np.floor(float(med[0])-1.5*h)))
    x2=min(int(frame_width),int(np.ceil(float(med[2])+1.5*h)))
    y1=max(0,int(np.floor(float(med[1])-.30*h)))
    y2=min(int(frame_height),int(np.ceil(float(med[3])+.30*h)))
    return x1,y1,x2,y2


def split_tracks_on_text_change(
    video_path,
    tracks,
    low_similarity=.65,
    stable_similarity=.75,
):
    """Split coarse tracks when the subtitle glyph pattern changes abruptly.

    A split requires stable text immediately before and after the change.
    One-frame occlusion therefore does not become a new subtitle segment.
    """
    tracks=list(tracks)
    if not tracks:
        return []

    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_by_id={}
    frame_to_tracks=defaultdict(list)
    for track in tracks:
        fs=track.sorted_frames()
        if len(fs)<4:
            continue
        crop_by_id[track.track_id]=_track_crop(
            track,width,height
        )
        for f in fs:
            frame_to_tracks[int(f)].append(track.track_id)

    masks=defaultdict(dict)
    max_frame=max(frame_to_tracks,default=-1)
    for fi in range(max_frame+1):
        ok,frame=cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"cannot decode frame {fi}: {video_path}")
        for tid in frame_to_tracks.get(fi,()):
            x1,y1,x2,y2=crop_by_id[tid]
            masks[tid][fi]=_text_core_mask(
                frame[y1:y2,x1:x2]
            )
    cap.release()

    next_id=max((t.track_id for t in tracks),default=0)+1
    out=[]
    for track in tracks:
        fs=track.sorted_frames()
        if len(fs)<4 or track.track_id not in masks:
            out.append(track)
            continue

        boundaries=[]
        tm=masks[track.track_id]
        for i in range(2,len(fs)-1):
            prevprev,prev,cur,nxt=fs[i-2],fs[i-1],fs[i],fs[i+1]
            if not (
                prevprev+1==prev
                and prev+1==cur
                and cur+1==nxt
            ):
                continue
            before=_mask_similarity(tm[prevprev],tm[prev])
            change=_mask_similarity(tm[prev],tm[cur])
            after=_mask_similarity(tm[cur],tm[nxt])
            if (
                before>=float(stable_similarity)
                and change<float(low_similarity)
                and after>=float(stable_similarity)
            ):
                boundaries.append(cur)

        if not boundaries:
            out.append(track)
            continue

        starts=[fs[0]]+boundaries
        ends=[b-1 for b in boundaries]+[fs[-1]]
        first=True
        for start,end in zip(starts,ends):
            chunk_fs=[f for f in fs if start<=f<=end]
            if not chunk_fs:
                continue
            is_first=first
            tid=track.track_id if is_first else next_id
            if not is_first:
                next_id+=1
            first=False
            chunk=SubtitleTrack(
                tid,
                confirmed=track.confirmed,
                language_status=track.language_status,
                content_boundary_before=(not is_first),
            )
            for f in chunk_fs:
                chunk.observations[f]=track.observations[f]
            out.append(chunk)

    out.sort(
        key=lambda t:(
            min(t.sorted_frames()) if t.sorted_frames() else 10**12,
            t.track_id,
        )
    )
    return out
