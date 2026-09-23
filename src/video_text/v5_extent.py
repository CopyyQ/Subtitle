from __future__ import annotations

import math
import cv2
import numpy as np

from .v5_slots import _text_core_mask


def _runs(active):
    out=[]
    start=None
    vals=active.tolist()+[False]
    for i,v in enumerate(vals):
        if v and start is None:
            start=i
        elif not v and start is not None:
            out.append([start,i-1])
            start=None
    return out


def _merge_close(runs,max_gap):
    if not runs:
        return []
    out=[runs[0][:]]
    for a,b in runs[1:]:
        if a-out[-1][1]-1 <= max_gap:
            out[-1][1]=b
        else:
            out.append([a,b])
    return out




def _glyph_like_run(width,area,h,yspan):
    # Normal Chinese glyph: meaningful width and vertical stroke coverage.
    # Thin horizontal glyphs such as "一" are explicitly preserved by the
    # wide-but-thin branch. Small enhanced speckles satisfy neither branch.
    h=max(1.0,float(h))
    width=float(width)
    area=float(area)
    yspan=float(yspan)
    normal=(
        width >= .15*h
        and yspan >= .30*h
        and area >= .008*h*h
    )
    thin_horizontal=(
        width >= .55*h
        and yspan >= .06*h
        and area >= .008*h*h
    )
    compact_punctuation=(
        area >= .03*h*h
        and yspan >= .18*h
    )
    return bool(normal or thin_horizontal or compact_punctuation)


def _frame_runs(mask,h):
    col_counts=mask.sum(axis=0)
    min_col=max(1,int(round(mask.shape[0]*.04)))
    active=col_counts>=min_col
    raw=_merge_close(_runs(active),max_gap=2)
    out=[]
    for a,b in raw:
        width=b-a+1
        area=int(mask[:,a:b+1].sum())
        ys,xs=np.where(mask[:,a:b+1]>0)
        if len(ys)==0:
            continue
        yspan=int(ys.max()-ys.min()+1)
        if not _glyph_like_run(width,area,h,yspan):
            continue
        out.append((a,b,area,width,yspan))
    return out


def _supported_edge_boundary(values,h,min_support=2):
    if len(values)<int(min_support):
        return None
    tolerance=max(2.0,.25*float(h))
    vals=sorted(float(v) for v in values)
    clusters=[]
    for value in vals:
        placed=False
        for cluster in clusters:
            center=float(np.median(cluster))
            if abs(value-center)<=tolerance:
                cluster.append(value)
                placed=True
                break
        if not placed:
            clusters.append([value])
    clusters=[c for c in clusters if len(c)>=int(min_support)]
    if not clusters:
        return None
    clusters.sort(key=lambda c:(len(c),-np.std(c)),reverse=True)
    return float(np.median(clusters[0]))


def _recover_occluded_edges(masks,left,right,h,max_gap):
    # A damaged/occluded edge glyph may disappear from the temporal vote even
    # though it is clearly visible in a few frames. Recover only glyph-sized
    # runs adjacent to the stable core, and require the same outer boundary in
    # at least two frames.
    left_values=[]
    right_values=[]
    min_extension=.08*float(h)
    for mask in masks:
        for a,b,area,width,yspan in _frame_runs(mask,h):
            # A standalone glyph can justify a large recovery when it is
            # temporarily occluded. A very wide merged/background component
            # may only make a small edge correction.
            max_extension=(
                .90*float(h)
                if float(width) <= 1.05*float(h)
                else .20*float(h)
            )

            left_extension=float(left-a)
            if min_extension <= left_extension <= max_extension:
                gap=max(0.0,float(left-b-1))
                if gap <= float(max_gap):
                    left_values.append(float(a))

            right_extension=float((b+1)-right)
            if min_extension <= right_extension <= max_extension:
                gap=max(0.0,float(a-right))
                if gap <= float(max_gap):
                    right_values.append(float(b+1))

    left_boundary=_supported_edge_boundary(left_values,h)
    right_boundary=_supported_edge_boundary(right_values,h)
    if left_boundary is not None:
        left=min(float(left),left_boundary)
    if right_boundary is not None:
        right=max(float(right),right_boundary)
    return float(left),float(right)





def _supported_normal_edge(
    masks,
    coarse_edge,
    h,
    direction,
    min_support=2,
    max_offset_ratio=.65,
):
    values=[]
    h=max(1.0,float(h))
    max_offset=float(max_offset_ratio)*h
    for mask in masks:
        for a,b,area,width,yspan in _frame_runs(mask,h):
            normal=(
                float(width) >= .15*h
                and float(width) <= 1.70*h
                and float(yspan) >= .30*h
                and float(area) >= .008*h*h
            )
            if not normal:
                continue
            if direction=="left":
                offset=float(coarse_edge)-float(a)
                if .08*h <= offset <= max_offset:
                    values.append(float(a))
            elif direction=="right":
                offset=float(b+1)-float(coarse_edge)
                if .08*h <= offset <= max_offset:
                    values.append(float(b+1))
    return _supported_edge_boundary(
        values,
        h,
        min_support=min_support,
    )

def _supported_thin_edge(masks,coarse_edge,h,direction,min_support=2):
    values=[]
    h=max(1.0,float(h))
    for mask in masks:
        for a,b,area,width,yspan in _frame_runs(mask,h):
            thin=(
                float(width) >= .55*h
                and float(yspan) <= .22*h
            )
            if not thin:
                continue
            if direction=="left" and float(a) < float(coarse_edge)-.08*h:
                values.append(float(a))
            elif direction=="right" and float(b+1) > float(coarse_edge)+.08*h:
                values.append(float(b+1))
    return _supported_edge_boundary(values,h,min_support=min_support)

def consensus_slot_bbox(
    frames,
    slot,
    support_ratio=.43,
    max_char_gap_ratio=.45,
    search_half_width_heights=8.0,
    outline_pad_ratio=.10,
    min_outline_pad=3,
):
    """Return one canonical x-extent for a locked vertical subtitle slot.

    A pixel must look like white-core + dark-outline in several frames before
    it can affect the canonical extent. This prevents a single bad frame from
    stretching the entire subtitle track.
    """
    frames=list(frames)
    if not frames:
        return None

    fh,fw=frames[0].shape[:2]
    y1=max(0,min(fh,int(round(slot.y1))))
    y2=max(0,min(fh,int(round(slot.y2))))
    if y2<=y1:
        return None

    h=max(1.0,float(slot.expected_height))
    cx=float(slot.expected_x_center)
    half=max(40,int(round(float(search_half_width_heights)*h)))
    x1=max(0,int(np.floor(cx-half)))
    x2=min(fw,int(np.ceil(cx+half)))
    if x2<=x1:
        return None

    masks=[]
    for frame in frames:
        crop=frame[y1:y2,x1:x2]
        masks.append(_text_core_mask(crop))
    stack=np.stack(masks,axis=0)
    need=max(2,int(math.ceil(len(masks)*float(support_ratio))))
    vote=(stack.sum(axis=0)>=need).astype(np.uint8)

    # A real glyph should occupy multiple vertical pixels; this also removes
    # isolated compression speckles that happen to survive temporal voting.
    col_counts=vote.sum(axis=0)
    min_col=max(1,int(round((y2-y1)*.04)))
    active=col_counts>=min_col
    raw=_runs(active)
    if not raw:
        return None

    # First close only tiny intra-glyph holes. We later chain whole glyph runs
    # outward from the line center using a font-height-scaled character gap.
    raw=_merge_close(raw,max_gap=2)
    runs=[]
    for a,b in raw:
        width=b-a+1
        band=vote[:,a:b+1]
        area=int(band.sum())
        ys,xs=np.where(band>0)
        if len(ys)==0:
            continue
        yspan=int(ys.max()-ys.min()+1)
        if not _glyph_like_run(width,area,h,yspan):
            continue
        runs.append([a,b])
    if not runs:
        return None

    center_local=cx-x1
    # Prefer a run covering the learned center. Otherwise choose the nearest
    # run; short centered second lines and one-character subtitles are valid.
    seed=min(
        range(len(runs)),
        key=lambda i: (
            0 if runs[i][0] <= center_local <= runs[i][1] else 1,
            min(abs(center_local-runs[i][0]),abs(center_local-runs[i][1])),
        ),
    )

    max_gap=float(max_char_gap_ratio)*h
    left=right=seed
    while left>0:
        gap=runs[left][0]-runs[left-1][1]-1
        if gap>max_gap:
            break
        left-=1
    while right+1<len(runs):
        gap=runs[right+1][0]-runs[right][1]-1
        if gap>max_gap:
            break
        right+=1

    local_left=float(runs[left][0])
    local_right=float(runs[right][1]+1)
    local_left,local_right=_recover_occluded_edges(
        masks,
        local_left,
        local_right,
        h,
        max_gap=float(max_char_gap_ratio)*h,
    )

    rx1=float(x1+local_left)
    rx2=float(x1+local_right)
    width=rx2-rx1
    if width < .18*h:
        return None

    pad=max(int(min_outline_pad),int(round(h*float(outline_pad_ratio))))
    rx1=max(0.0,rx1-pad)
    rx2=min(float(fw),rx2+pad)

    # Enhancement may attach a bright/dark scene structure to the subtitle.
    # A FAST-backed slot therefore provides a conservative horizontal guard.
    # Stable thin horizontal glyphs (e.g. 一 / ----) are the exception because
    # scene-text detectors commonly miss them.
    guard_margin=.10*h
    if slot.expected_x1 is not None:
        min_allowed=max(0.0,float(slot.expected_x1)-guard_margin)
        if rx1 < min_allowed:
            coarse_local=float(slot.expected_x1)-float(x1)
            thin_left=_supported_thin_edge(
                masks,
                coarse_local,
                h,
                direction="left",
            )
            if thin_left is None:
                rx1=min_allowed
            else:
                rx1=max(0.0,float(x1)+thin_left-pad)

    if slot.expected_x2 is not None:
        max_allowed=min(float(fw),float(slot.expected_x2)+guard_margin)
        if rx2 > max_allowed:
            thin_right=_supported_thin_edge(
                masks,
                float(slot.expected_x2)-float(x1),
                h,
                direction="right",
            )
            if thin_right is None:
                rx2=max_allowed
            else:
                rx2=min(float(fw),float(x1)+thin_right+pad)

    return np.array([rx1,float(y1),rx2,float(y2)],np.float32)
