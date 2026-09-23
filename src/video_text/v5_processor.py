from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from .association import AssociationConfig, build_provisional_tracks
from .lifecycle import split_tracks_on_geometry
from .smoothing import expand_bbox_for_outline
from .types import SubtitleTrack, TrackObservation
from .v5_extent import consensus_slot_bbox
from .v5_content_split import split_tracks_on_text_change
from .v5_segments import build_subtitle_segments
from .v5_slots import bootstrap_line_slots
from .v5_weak_birth import SlotPrior


def _clip_box(box,width,height):
    b=np.asarray(box,dtype=np.float32).copy()
    b[0]=max(0.0,min(float(width),float(b[0])))
    b[2]=max(0.0,min(float(width),float(b[2])))
    b[1]=max(0.0,min(float(height),float(b[1])))
    b[3]=max(0.0,min(float(height),float(b[3])))
    return b


def _sample_indices(start,end,count):
    start=int(start); end=int(end); count=max(1,int(count))
    if end<=start or count==1:
        return [start]
    n=end-start+1
    if n<=count:
        return list(range(start,end+1))
    vals=np.linspace(start,end,count)
    return sorted({int(round(x)) for x in vals})


def _read_frames(video_path,indices):
    indices=sorted(set(int(x) for x in indices))
    if not indices:
        return {}
    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    out={}
    for fi in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES,fi)
        ok,frame=cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"cannot decode frame {fi}: {video_path}")
        out[fi]=frame
    cap.release()
    return out


def _fallback_slots(segment,frame_width,frame_height):
    b=np.asarray(segment.median_bbox,dtype=np.float32)
    h=max(1.0,float(b[3]-b[1]))
    box=expand_bbox_for_outline(
        b,
        frame_width=frame_width,
        frame_height=frame_height,
        pad_ratio=.08,
        min_pad=3,
    )
    return [SlotPrior(
        y1=int(round(box[1])),
        y2=int(round(box[3])),
        expected_x_center=float((box[0]+box[2])*.5),
        expected_height=float(max(1.0,box[3]-box[1])),
        expected_x1=float(box[0]),
        expected_x2=float(box[2]),
    )]


def _fallback_bbox(segment,slot,frame_width,frame_height):
    b=np.asarray(segment.median_bbox,dtype=np.float32).copy()
    # Preserve the locked vertical slot, but use conservative median FAST x
    # when pixel consensus cannot form a reliable text extent.
    b[1]=float(slot.y1)
    b[3]=float(slot.y2)
    return expand_bbox_for_outline(
        _clip_box(b,frame_width,frame_height),
        frame_width=frame_width,
        frame_height=frame_height,
        pad_ratio=.08,
        min_pad=3,
    )




def _track_median_box(track):
    fs=track.sorted_frames()
    if not fs:
        return None
    return np.median(
        np.stack([track.observations[f].bbox for f in fs],axis=0),
        axis=0,
    ).astype(np.float32)


def coarse_line_slots_from_tracks(
    segment,
    trackmap,
    frame_width,
    frame_height,
    min_overlap_ratio=.55,
):
    """Return two reliable line slots when coarse FAST already separates rows.

    Only temporally concurrent tracks can define a two-row layout. Sequential
    same-lane subtitle fragments must never be interpreted as two lines.
    """
    candidates=[]
    ids=[tid for tid in segment.track_ids if tid in trackmap]
    for i,tid_a in enumerate(ids):
        ta=trackmap[tid_a]
        fa=set(ta.sorted_frames())
        ba=_track_median_box(ta)
        if ba is None or not fa:
            continue
        ha=max(1.0,float(ba[3]-ba[1]))
        ca=(float(ba[1])+float(ba[3]))*.5
        for tid_b in ids[i+1:]:
            tb=trackmap[tid_b]
            fb=set(tb.sorted_frames())
            bb=_track_median_box(tb)
            if bb is None or not fb:
                continue
            overlap=len(fa & fb)
            ratio=overlap/max(1,min(len(fa),len(fb)))
            if ratio < float(min_overlap_ratio):
                continue
            hb=max(1.0,float(bb[3]-bb[1]))
            cb=(float(bb[1])+float(bb[3]))*.5
            sep=abs(ca-cb)/max(ha,hb)
            if not (.45 <= sep <= 1.65):
                continue
            candidates.append((overlap,ratio,ba,bb))
    if not candidates:
        return []

    _,_,a,b=max(candidates,key=lambda x:(x[0],x[1]))
    rows=sorted([a,b],key=lambda box:(float(box[1])+float(box[3]))*.5)
    slots=[]
    for box in rows:
        h=max(1.0,float(box[3]-box[1]))
        pad=max(3,int(round(h*.08)))
        y1=max(0,int(np.floor(float(box[1])))-pad)
        y2=min(int(frame_height),int(np.ceil(float(box[3])))+pad)
        slots.append(SlotPrior(
            y1=y1,
            y2=y2,
            expected_x_center=float((box[0]+box[2])*.5),
            expected_height=float(max(1,y2-y1)),
            expected_x1=float(box[0]),
            expected_x2=float(box[2]),
        ))
    return slots




def coarse_supported_slots_from_tracks(
    segment,
    trackmap,
    frame_width,
    frame_height,
):
    rows=[]
    for tid in segment.track_ids:
        track=trackmap.get(tid)
        if track is None:
            continue
        box=_track_median_box(track)
        if box is None:
            continue
        h=max(1.0,float(box[3]-box[1]))
        fs=track.sorted_frames()
        observed_x1=min(
            float(track.observations[f].bbox[0])
            for f in fs
        )
        rows.append({
            "box":box,
            "cy":float((box[1]+box[3])*.5),
            "h":h,
            "observed_x1":observed_x1,
        })
    if not rows:
        return []

    rows=sorted(rows,key=lambda r:r["cy"])
    groups=[]
    for row in rows:
        placed=False
        for group in groups:
            med_cy=float(np.median([x["cy"] for x in group]))
            med_h=float(np.median([x["h"] for x in group]))
            if abs(row["cy"]-med_cy) <= .35*max(row["h"],med_h):
                group.append(row)
                placed=True
                break
        if not placed:
            groups.append([row])

    slots=[]
    for group in groups:
        boxes=np.stack([x["box"] for x in group],axis=0)
        med=np.median(boxes,axis=0).astype(np.float32)
        h=max(1.0,float(med[3]-med[1]))
        pad=max(3,int(round(h*.08)))
        y1=max(0,int(np.floor(float(med[1])))-pad)
        y2=min(int(frame_height),int(np.ceil(float(med[3])))+pad)
        slots.append(SlotPrior(
            y1=y1,
            y2=y2,
            expected_x_center=float((med[0]+med[2])*.5),
            expected_height=float(max(1,y2-y1)),
            expected_x1=float(min(x["observed_x1"] for x in group)),
            expected_x2=float(med[2]),
        ))
    slots.sort(key=lambda slot:(slot.y1+slot.y2)*.5)
    return slots


def merge_layout_slots(coarse_slots,pixel_slots,max_slots=2):
    selected=[(slot,True) for slot in coarse_slots]
    for slot in pixel_slots:
        scy=(float(slot.y1)+float(slot.y2))*.5
        sh=max(1.0,float(slot.y2-slot.y1))
        duplicate=False
        for existing,_ in selected:
            ecy=(float(existing.y1)+float(existing.y2))*.5
            eh=max(1.0,float(existing.y2-existing.y1))
            if abs(scy-ecy) <= .35*max(sh,eh):
                duplicate=True
                break
        if not duplicate:
            selected.append((slot,False))
    selected.sort(key=lambda item:(item[0].y1+item[0].y2)*.5)
    if len(selected)>int(max_slots):
        selected=sorted(
            selected,
            key=lambda item:(0 if item[1] else 1,(item[0].y1+item[0].y2)*.5),
        )[:int(max_slots)]
        selected.sort(key=lambda item:(item[0].y1+item[0].y2)*.5)
    return selected

def slot_has_coarse_support(slot,segment,trackmap):
    scy=(float(slot.y1)+float(slot.y2))*.5
    sh=max(1.0,float(slot.y2-slot.y1))
    for tid in segment.track_ids:
        track=trackmap.get(tid)
        if track is None:
            continue
        box=_track_median_box(track)
        if box is None:
            continue
        th=max(1.0,float(box[3]-box[1]))
        tcy=(float(box[1])+float(box[3]))*.5
        if abs(tcy-scy) <= .35*max(sh,th):
            return True
    return False



def analysis_slots_for_layout(slots):
    """Create non-overlapping Y strips used only for X-extent analysis."""
    slots=list(slots)
    if len(slots)!=2:
        return slots
    ordered=sorted(slots,key=lambda x:(float(x.y1)+float(x.y2))*.5)
    top,bottom=ordered
    top_c=(float(top.y1)+float(top.y2))*.5
    bot_c=(float(bottom.y1)+float(bottom.y2))*.5
    separator=int(round((top_c+bot_c)*.5))
    top_a=SlotPrior(
        y1=int(top.y1),
        y2=max(int(top.y1)+1,min(int(top.y2),separator)),
        expected_x_center=float(top.expected_x_center),
        expected_height=float(top.expected_height),
        expected_x1=top.expected_x1,
        expected_x2=top.expected_x2,
    )
    bottom_a=SlotPrior(
        y1=min(int(bottom.y2)-1,max(int(bottom.y1),separator)),
        y2=int(bottom.y2),
        expected_x_center=float(bottom.expected_x_center),
        expected_height=float(bottom.expected_height),
        expected_x1=bottom.expected_x1,
        expected_x2=bottom.expected_x2,
    )
    return [top_a,bottom_a]

def build_v5_strong_tracks(
    video_path,
    frames,
    frame_width,
    frame_height,
    max_internal_gap=2,
    bootstrap_count=7,
    sample_count=9,
):
    """Create zero-jitter tracks from FAST timing + segment-level line slots.

    FAST supplies coarse temporal segments only. The vertical layout is locked
    once per segment from several bootstrap frames, and x extents are computed
    by temporal consensus instead of mutating each frame independently.
    """
    cfg=AssociationConfig(max_frame_gap=int(max_internal_gap)+1)
    provisional=build_provisional_tracks(frames,cfg)
    provisional=split_tracks_on_geometry(
        provisional,
        max_internal_gap=int(max_internal_gap),
    )
    coarse_track_count=len(provisional)
    provisional=split_tracks_on_text_change(
        video_path,
        provisional,
    )
    content_split_track_count=len(provisional)
    segments=build_subtitle_segments(provisional)
    trackmap={t.track_id:t for t in provisional}

    tracks=[]
    next_id=1
    two_line_segments=0
    fallback_slot_segments=0
    fallback_extent_count=0
    coarse_layout_segment_count=0
    pixel_only_slot_count=0

    for segment in segments:
        boot_idx=list(range(
            int(segment.start_frame),
            min(int(segment.end_frame)+1,int(segment.start_frame)+int(bootstrap_count)),
        ))
        sample_idx=_sample_indices(
            segment.start_frame,
            segment.end_frame,
            sample_count,
        )
        needed=sorted(set(boot_idx+sample_idx))
        frame_map=_read_frames(video_path,needed)

        search=_clip_box(
            segment.search_bbox,
            frame_width,
            frame_height,
        )
        boot_frames=[frame_map[i] for i in boot_idx if i in frame_map]

        coarse_slots=coarse_supported_slots_from_tracks(
            segment,
            trackmap,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        pixel_slots=bootstrap_line_slots(
            boot_frames,
            search_bbox=search,
        )
        merged=merge_layout_slots(
            coarse_slots,
            pixel_slots,
            max_slots=2,
        )
        if coarse_slots:
            coarse_layout_segment_count+=1
        if not merged:
            merged=[(slot,False) for slot in _fallback_slots(
                segment,frame_width,frame_height
            )]
            fallback_slot_segments+=1
        if len(merged)==2:
            two_line_segments+=1

        sample_frames=[frame_map[i] for i in sample_idx if i in frame_map]
        slots=[slot for slot,_ in merged]
        analysis_slots=analysis_slots_for_layout(slots)
        for (slot,coarse_supported),analysis_slot in zip(merged,analysis_slots):
            canonical=consensus_slot_bbox(sample_frames,analysis_slot)
            if canonical is None:
                canonical=_fallback_bbox(
                    segment,slot,frame_width,frame_height
                )
                fallback_extent_count+=1
            else:
                canonical=_clip_box(
                    canonical,frame_width,frame_height
                )
                # X is estimated from the isolated analysis strip, while Y
                # remains the original outline-safe locked slot.
                canonical[1]=float(slot.y1)
                canonical[3]=float(slot.y2)

            level="V5_SLOT" if coarse_supported else "V5_SLOT_PIXEL_ONLY"
            if not coarse_supported:
                pixel_only_slot_count+=1

            track=SubtitleTrack(next_id,confirmed=True)
            next_id+=1
            for fi in range(segment.start_frame,segment.end_frame+1):
                track.observations[int(fi)]=TrackObservation(
                    int(fi),
                    canonical.copy(),
                    None,
                    level,
                    reconstructed=False,
                )
            tracks.append(track)

    metrics={
        "segment_count":len(segments),
        "two_line_segment_count":two_line_segments,
        "fallback_slot_segment_count":fallback_slot_segments,
        "fallback_extent_count":fallback_extent_count,
        "coarse_layout_segment_count":coarse_layout_segment_count,
        "pixel_only_slot_count":pixel_only_slot_count,
        "strong_track_count":len(tracks),
        "coarse_track_count":coarse_track_count,
        "content_split_track_count":content_split_track_count,
    }
    return tracks,metrics


def learn_global_slot_priors(strong_tracks,frame_width):
    rows=[]
    for track in strong_tracks:
        fs=track.sorted_frames()
        if not fs:
            continue
        b=np.asarray(track.observations[fs[0]].bbox,dtype=np.float32)
        h=max(1.0,float(b[3]-b[1]))
        rows.append({
            "y1":float(b[1]),
            "y2":float(b[3]),
            "cy":float((b[1]+b[3])*.5),
            "cx":float((b[0]+b[2])*.5),
            "h":h,
        })
    if not rows:
        return []

    rows=sorted(rows,key=lambda r:r["cy"])
    clusters=[]
    for row in rows:
        best=None
        best_dist=None
        for i,cluster in enumerate(clusters):
            med_cy=float(np.median([x["cy"] for x in cluster]))
            med_h=float(np.median([x["h"] for x in cluster]))
            dist=abs(row["cy"]-med_cy)
            if dist <= .42*max(row["h"],med_h):
                if best_dist is None or dist<best_dist:
                    best=i
                    best_dist=dist
        if best is None:
            clusters.append([row])
        else:
            clusters[best].append(row)

    priors=[]
    for cluster in clusters:
        y1=int(round(float(np.median([x["y1"] for x in cluster]))))
        y2=int(round(float(np.median([x["y2"] for x in cluster]))))
        h=max(1.0,float(np.median([x["h"] for x in cluster])))
        cx=float(np.median([x["cx"] for x in cluster]))
        cx=max(0.0,min(float(frame_width),cx))
        priors.append(SlotPrior(y1,y2,cx,h))
    priors.sort(key=lambda p:(p.y1+p.y2)*.5)
    return priors


def _bbox_iou(a,b):
    x1=max(float(a[0]),float(b[0]))
    y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2]))
    y2=min(float(a[3]),float(b[3]))
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    aa=max(0.0,float(a[2]-a[0]))*max(0.0,float(a[3]-a[1]))
    bb=max(0.0,float(b[2]-b[0]))*max(0.0,float(b[3]-b[1]))
    den=aa+bb-inter
    return inter/den if den>0 else 0.0


def _records_to_weak_tracks(records_by_prior,strong_tracks,confirm_frames):
    next_id=max([t.track_id for t in strong_tracks],default=0)+1
    tracks=[]
    for records in records_by_prior:
        if not records:
            continue
        by_frame={}
        for row in records:
            by_frame[int(row.frame)]=row
        ordered=[by_frame[f] for f in sorted(by_frame)]
        groups=[]
        current=[]
        for row in ordered:
            if not current:
                current=[row]
                continue
            prev=current[-1]
            contiguous=row.frame==prev.frame+1
            similar=_bbox_iou(prev.bbox,row.bbox)>=.45
            if contiguous and similar:
                current.append(row)
            else:
                groups.append(current)
                current=[row]
        if current:
            groups.append(current)

        for group in groups:
            if len(group)<int(confirm_frames):
                continue
            track=SubtitleTrack(next_id,confirmed=True)
            next_id+=1
            canonical=np.median(
                np.stack([r.bbox for r in group],axis=0),
                axis=0,
            ).astype(np.float32)
            for row in group:
                track.observations[int(row.frame)]=TrackObservation(
                    int(row.frame),
                    canonical.copy(),
                    None,
                    "V5_WEAK",
                    reconstructed=False,
                )
            tracks.append(track)
    return tracks


def _suppress_duplicate_weak_tracks(tracks):
    keep=[]
    for track in sorted(tracks,key=lambda t:(min(t.sorted_frames()),-len(t.sorted_frames()))):
        fs=set(track.sorted_frames())
        if not fs:
            continue
        box=track.observations[min(fs)].bbox
        duplicate=False
        for kept in keep:
            kfs=set(kept.sorted_frames())
            overlap=len(fs & kfs)
            if overlap==0:
                continue
            kbox=kept.observations[min(kfs)].bbox
            temporal=overlap/max(1,min(len(fs),len(kfs)))
            if temporal>=.75 and _bbox_iou(box,kbox)>=.65:
                duplicate=True
                break
        if not duplicate:
            keep.append(track)
    return keep


def discover_v5_weak_tracks(
    video_path,
    strong_tracks,
    priors,
    frame_count,
    frame_width,
    frame_height,
    confirm_frames=3,
):
    from .v5_weak_birth import WeakBirthTracker, detect_weak_text_in_slot

    priors=list(priors)
    if not priors:
        return [],{"weak_track_count":0,"weak_frame_count":0}

    covered=set()
    for track in strong_tracks:
        covered.update(track.sorted_frames())

    trackers=[
        WeakBirthTracker(confirm_frames=confirm_frames,max_gap=0)
        for _ in priors
    ]
    emitted=[[] for _ in priors]

    cap=cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    for fi in range(int(frame_count)):
        ok,frame=cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"cannot decode frame {fi}: {video_path}")

        if fi in covered:
            for tracker in trackers:
                tracker.update(fi,None)
            continue

        for i,(prior,tracker) in enumerate(zip(priors,trackers)):
            box=detect_weak_text_in_slot(frame,prior)
            rows=tracker.update(fi,box)
            emitted[i].extend(rows)

    cap.release()
    weak=_records_to_weak_tracks(
        emitted,
        strong_tracks=strong_tracks,
        confirm_frames=confirm_frames,
    )
    weak=_suppress_duplicate_weak_tracks(weak)
    metrics={
        "weak_track_count":len(weak),
        "weak_frame_count":sum(len(t.observations) for t in weak),
    }
    return weak,metrics
