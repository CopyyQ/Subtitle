from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np

from .types import SubtitleTrack
from .v5_content_split import _mask_similarity, _track_crop
from .v5_slots import _text_core_mask


def _consecutive(frames):
    return all(b==a+1 for a,b in zip(frames,frames[1:]))


def _stable_similarity(frame_ids,masks):
    if len(frame_ids)<2:
        return 1.0
    vals=[
        _mask_similarity(masks[a],masks[b])
        for a,b in zip(frame_ids,frame_ids[1:])
    ]
    return float(np.median(vals)) if vals else 1.0


def _cross_similarity(before,after,masks):
    vals=[
        _mask_similarity(masks[a],masks[b])
        for a in before for b in after
    ]
    return float(np.median(vals)) if vals else 1.0

def _candidate_boundaries(
    fs,
    masks,
    window,
    change_threshold,
    stable_threshold,
    transition_tolerance=2,
):
    candidates=[]
    n=len(fs)
    for i in range(int(window),n-int(window)+1):
        before=fs[i-window:i]
        if len(before)<window or not _consecutive(before):
            continue
        pre_stable=_stable_similarity(before,masks)
        if pre_stable<float(stable_threshold):
            continue

        best=None
        for gap in range(int(transition_tolerance)+1):
            j=i+gap
            after=fs[j:j+window]
            if len(after)<window or not _consecutive(after):
                continue
            post_stable=_stable_similarity(after,masks)
            if post_stable<float(stable_threshold):
                continue
            cross=_cross_similarity(before,after,masks)
            adaptive=max(
                float(change_threshold),
                min(.72,.5*(pre_stable+post_stable)-.28),
            )
            if cross>=adaptive:
                continue
            item=(cross,fs[j],gap,pre_stable,post_stable)
            if best is None or item[0]<best[0]:
                best=item
        if best is not None:
            candidates.append(best)

    return candidates

def _suppress_close_candidates(candidates,window):
    if not candidates:
        return []
    ordered=sorted(candidates,key=lambda x:int(x[1]))
    groups=[]
    current=[ordered[0]]
    for row in ordered[1:]:
        if int(row[1])-int(current[-1][1]) <= int(window):
            current.append(row)
        else:
            groups.append(current)
            current=[row]
    groups.append(current)
    chosen=[
        min(group,key=lambda x:(float(x[0]),int(x[1])))
        for group in groups
    ]
    return sorted({int(x[1]) for x in chosen})


def split_tracks_on_text_change_windowed(
    video_path,
    tracks,
    window=3,
    change_threshold=.62,
    stable_threshold=.72,
):
    tracks=list(tracks)
    if not tracks:
        return []

    window=max(2,int(window))
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_by_id={}
    frame_to_tracks=defaultdict(list)
    for track in tracks:
        fs=track.sorted_frames()
        if len(fs)<2*window:
            continue
        crop_by_id[track.track_id]=_track_crop(track,width,height)
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
            masks[tid][fi]=_text_core_mask(frame[y1:y2,x1:x2])
    cap.release()

    next_id=max((t.track_id for t in tracks),default=0)+1
    out=[]
    for track in tracks:
        fs=track.sorted_frames()
        if len(fs)<2*window or track.track_id not in masks:
            out.append(track)
            continue

        candidates=_candidate_boundaries(
            fs,
            masks[track.track_id],
            window,
            change_threshold,
            stable_threshold,
        )
        boundaries=_suppress_close_candidates(candidates,window)
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
