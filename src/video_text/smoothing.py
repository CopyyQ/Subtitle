from __future__ import annotations
import copy
import numpy as np


def smooth_track(track,window=5):
    if window<1 or window%2==0:
        raise ValueError("smoothing window must be a positive odd integer")
    out=copy.deepcopy(track)
    frames=sorted(track.observations)
    if len(frames)<3:
        return out
    half=window//2
    raw=[track.observations[f].bbox.copy() for f in frames]
    for i,f in enumerate(frames):
        lo=max(0,i-half)
        hi=min(len(frames),i+half+1)
        vals=np.stack(raw[lo:hi],axis=0)
        out.observations[f].bbox=np.median(vals,axis=0).astype(np.float32)
    return out


def smooth_tracks(tracks,window=5):
    return [smooth_track(t,window=window) for t in tracks]


def robust_track_envelope(track):
    frames=track.sorted_frames()
    if not frames:
        return None
    observed=[
        track.observations[f].bbox
        for f in frames
        if not track.observations[f].reconstructed
    ]
    if not observed:
        observed=[track.observations[f].bbox for f in frames]
    arr=np.stack(observed,axis=0).astype(np.float32)
    n=len(arr)
    rank=1 if n>=5 else 0
    x1=np.sort(arr[:,0])[rank]
    y1=np.sort(arr[:,1])[rank]
    x2=np.sort(arr[:,2])[-(rank+1)]
    y2=np.sort(arr[:,3])[-(rank+1)]
    return np.array([x1,y1,x2,y2],np.float32)


def expand_bbox_for_outline(
    bbox,
    frame_width,
    frame_height,
    pad_ratio=.08,
    min_pad=3,
):
    box=np.asarray(bbox,dtype=np.float32).copy()
    h=max(1.0,float(box[3]-box[1]))
    pad=max(int(min_pad),int(round(h*float(pad_ratio))))
    box[0]=max(0.0,float(box[0])-pad)
    box[1]=max(0.0,float(box[1])-pad)
    box[2]=min(float(frame_width),float(box[2])+pad)
    box[3]=min(float(frame_height),float(box[3])+pad)
    return box.astype(np.float32)


def synchronize_track_bbox(
    track,
    frame_width=None,
    frame_height=None,
    outline_pad_ratio=0.0,
    min_outline_pad=0,
):
    out=copy.deepcopy(track)
    frames=out.sorted_frames()
    if not frames:
        return out
    canonical=robust_track_envelope(out)
    if (
        frame_width is not None
        and frame_height is not None
        and (outline_pad_ratio>0 or min_outline_pad>0)
    ):
        canonical=expand_bbox_for_outline(
            canonical,
            frame_width=frame_width,
            frame_height=frame_height,
            pad_ratio=outline_pad_ratio,
            min_pad=min_outline_pad,
        )
    for f in frames:
        out.observations[f].bbox=canonical.copy()
    return out


def synchronize_tracks(
    tracks,
    frame_width=None,
    frame_height=None,
    outline_pad_ratio=0.0,
    min_outline_pad=0,
):
    return [
        synchronize_track_bbox(
            t,
            frame_width=frame_width,
            frame_height=frame_height,
            outline_pad_ratio=outline_pad_ratio,
            min_outline_pad=min_outline_pad,
        )
        for t in tracks
    ]
