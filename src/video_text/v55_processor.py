from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .association import AssociationConfig, build_provisional_tracks
from .lifecycle import split_tracks_on_geometry
from .types import SubtitleTrack, TrackObservation
from .text_enhancement import white_black_text_mask
from .v5_content_split import _track_crop
from .v5_extent import consensus_slot_bbox
from .v5_processor import (
    _clip_box,
    _fallback_bbox,
    _fallback_slots,
    _read_frames,
    _records_to_weak_tracks,
    _sample_indices,
    _suppress_duplicate_weak_tracks,
    analysis_slots_for_layout,
    coarse_supported_slots_from_tracks,
    merge_layout_slots,
)
from .v5_segments import build_subtitle_segments
from .v5_slots import bootstrap_line_slots, _text_core_mask
from .v55_content_split import split_tracks_on_text_change_windowed
from .v55_geometry import (
    enforce_non_overlapping_lines,
    line_overlap_area,
    refine_line_bbox,
    smooth_line_geometry,
)
from .v5_weak_birth import (
    SlotPrior,
    WeakBirthTracker,
    detect_weak_text_in_slot,
)


@dataclass(slots=True)
class V55TrackBundle:
    tracks: list[SubtitleTrack]
    identity: dict[int,tuple[int,int]]
    metrics: dict


@dataclass(slots=True)
class SlotCoverage:
    occupied: set[tuple[int,int]] = field(default_factory=set)

    def add(self,frame,slot_index):
        self.occupied.add((int(frame),int(slot_index)))

    def is_occupied(self,frame,slot_index):
        return (int(frame),int(slot_index)) in self.occupied

def _nearest_slot_index(box,priors,max_center_ratio=.45):
    if not priors:
        return None
    b=np.asarray(box,dtype=np.float32)
    h=max(1.0,float(b[3]-b[1]))
    cy=.5*(float(b[1])+float(b[3]))
    best=None
    for i,prior in enumerate(priors):
        ph=max(1.0,float(prior.expected_height))
        pcy=.5*(float(prior.y1)+float(prior.y2))
        dist=abs(cy-pcy)
        gate=float(max_center_ratio)*max(h,ph)
        if dist>gate:
            continue
        key=(dist,i)
        if best is None or key<best[0]:
            best=(key,i)
    return None if best is None else int(best[1])


def _build_slot_coverage(strong_tracks,priors):
    coverage=SlotCoverage()
    for track in strong_tracks:
        for fi in track.sorted_frames():
            box=track.observations[fi].bbox
            idx=_nearest_slot_index(box,priors)
            if idx is not None:
                coverage.add(fi,idx)
    return coverage

def _count_cross_slot_recoveries(weak,coverage,priors):
    count=0
    for track in weak:
        fs=track.sorted_frames()
        if not fs:
            continue
        slot=_nearest_slot_index(track.observations[fs[0]].bbox,priors)
        if slot is None:
            continue
        recovered=False
        for fi in fs:
            for other in range(len(priors)):
                if other==slot:
                    continue
                if coverage.is_occupied(fi,other):
                    recovered=True
                    break
            if recovered:
                break
        if recovered:
            count+=1
    return count


def _slot_geometry_matches(
    box,
    prior,
    *,
    max_center_offset_heights=.35,
    min_height_ratio=.70,
    max_height_ratio=1.20,
):
    b=np.asarray(box,dtype=np.float32)
    ph=max(1.0,float(prior.expected_height))
    bh=max(0.0,float(b[3]-b[1]))
    cx=.5*(float(b[0])+float(b[2]))
    if abs(cx-float(prior.expected_x_center)) > float(max_center_offset_heights)*ph:
        return False
    ratio=bh/ph
    return float(min_height_ratio) <= ratio <= float(max_height_ratio)


def _weak_birth_visual_gate(
    frame,
    box,
    prior,
    *,
    min_mask_density=.40,
):
    if box is None or not _slot_geometry_matches(box,prior):
        return None
    h,w=frame.shape[:2]
    b=np.asarray(box,dtype=np.float32)
    x1=max(0,int(np.floor(float(b[0]))))
    y1=max(0,int(np.floor(float(b[1]))))
    x2=min(w,int(np.ceil(float(b[2]))))
    y2=min(h,int(np.ceil(float(b[3]))))
    if x2<=x1 or y2<=y1:
        return None
    mask=white_black_text_mask(frame[y1:y2,x1:x2])
    if mask.size==0 or float(np.mean(mask>0)) < float(min_mask_density):
        return None
    return b


def filter_v55_pixel_only_tracks(tracks,priors):
    priors=list(priors)
    kept=[]
    rejected=0
    for track in tracks:
        fs=track.sorted_frames()
        if not fs:
            rejected+=1
            continue
        boxes=np.stack([track.observations[fi].bbox for fi in fs],axis=0)
        box=np.median(boxes,axis=0).astype(np.float32)
        slot=_nearest_slot_index(box,priors)
        if slot is None or not _slot_geometry_matches(box,priors[slot]):
            rejected+=1
            continue
        kept.append(track)
    return kept,rejected


def discover_v55_weak_tracks(
    video_path,
    strong_tracks,
    priors,
    frame_count,
    frame_width,
    frame_height,
    confirm_frames=3,
):
    del frame_width,frame_height
    priors=list(priors)
    if not priors:
        return [],{
            "weak_track_count":0,
            "weak_frame_count":0,
            "slot_weak_recovery_count":0,
        }

    coverage=_build_slot_coverage(strong_tracks,priors)
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
        for i,(prior,tracker) in enumerate(zip(priors,trackers)):
            if coverage.is_occupied(fi,i):
                tracker.update(fi,None)
                continue
            box=detect_weak_text_in_slot(frame,prior)
            box=_weak_birth_visual_gate(frame,box,prior)
            emitted[i].extend(tracker.update(fi,box))

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
        "slot_weak_recovery_count":_count_cross_slot_recoveries(
            weak,coverage,priors
        ),
    }
    return weak,metrics


def _edge_motion_metrics(tracks):
    vals=[]
    for track in tracks:
        fs=track.sorted_frames()
        for a,b in zip(fs,fs[1:]):
            if b!=a+1:
                continue
            ba=track.observations[a].bbox
            bb=track.observations[b].bbox
            vals.extend(np.abs(bb-ba).astype(np.float32).tolist())
    if not vals:
        return 0.0,0.0
    return float(np.median(vals)),float(np.percentile(vals,95))


def _mask_bbox_area(frame,box):
    h,w=frame.shape[:2]
    x1=max(0,int(np.floor(float(box[0]))))
    y1=max(0,int(np.floor(float(box[1]))))
    x2=min(w,int(np.ceil(float(box[2]))))
    y2=min(h,int(np.ceil(float(box[3]))))
    if x2<=x1 or y2<=y1:
        return None
    mask=_text_core_mask(frame[y1:y2,x1:x2])
    ys,xs=np.where(mask>0)
    if len(xs)<4:
        return None
    mw=float(xs.max()-xs.min()+1)
    mh=float(ys.max()-ys.min()+1)
    return max(1.0,mw*mh)


def build_v55_strong_tracks(
    video_path,
    frames,
    frame_width,
    frame_height,
    max_internal_gap=2,
    bootstrap_count=7,
    sample_count=9,
    refinement_window=5,
):
    cfg=AssociationConfig(max_frame_gap=int(max_internal_gap)+1)
    provisional=build_provisional_tracks(frames,cfg)
    provisional=split_tracks_on_geometry(
        provisional,
        max_internal_gap=int(max_internal_gap),
    )
    coarse_track_count=len(provisional)
    provisional=split_tracks_on_text_change_windowed(
        video_path,
        provisional,
    )
    content_split_track_count=len(provisional)
    segments=build_subtitle_segments(provisional)
    trackmap={t.track_id:t for t in provisional}

    tracks=[]
    identity={}
    next_track_id=1
    next_subtitle_id=1
    two_line_segments=0
    fallback_slot_segments=0
    fallback_extent_count=0
    coarse_layout_segment_count=0
    pixel_only_slot_count=0
    refinement_applied=0
    refinement_anchor_fallback=0
    overlap_pixels=0.0
    overlap_frames=0
    excess_samples=[]


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
        all_idx=list(range(int(segment.start_frame),int(segment.end_frame)+1))
        needed=sorted(set(boot_idx+sample_idx+all_idx))
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
            merged=[
                (slot,False)
                for slot in _fallback_slots(segment,frame_width,frame_height)
            ]
            fallback_slot_segments+=1
        if len(merged)==2:
            two_line_segments+=1

        slots=[slot for slot,_ in merged]
        analysis_slots=analysis_slots_for_layout(slots)
        sample_frames=[frame_map[i] for i in sample_idx if i in frame_map]
        per_line_boxes=[]
        per_line_levels=[]
        per_line_geometries=[]

        for (slot,coarse_supported),analysis_slot in zip(merged,analysis_slots):
            anchor=consensus_slot_bbox(
                sample_frames,
                analysis_slot,
                outline_pad_ratio=.02,
                min_outline_pad=1,
            )
            if anchor is None:
                anchor=_fallback_bbox(
                    segment,slot,frame_width,frame_height
                )
                fallback_extent_count+=1
            else:
                anchor=_clip_box(anchor,frame_width,frame_height)
            # V5.5 uses the locked slot only as an anchor prior. Final Y is
            # measured from glyph evidence per frame rather than forced back
            # to the padded slot envelope.
            geometries=[]
            for fi in all_idx:
                g=refine_line_bbox(
                    frame_map[fi],
                    anchor,
                    slot,
                    pad_px=max(2,int(round(float(slot.expected_height)*.04))),
                )
                geometries.append(g)
                if g.used_anchor:
                    refinement_anchor_fallback+=1
                else:
                    refinement_applied+=1
            smoothed=smooth_line_geometry(
                geometries,
                anchor,
                window=refinement_window,
                max_step_px=2.0,
            )
            per_line_boxes.append(smoothed)
            per_line_geometries.append(geometries)
            level="V55_LINE" if coarse_supported else "V55_LINE_PIXEL_ONLY"
            if not coarse_supported:
                pixel_only_slot_count+=1
            per_line_levels.append(level)

        # Enforce the no-overlap invariant independently on every frame.
        for k,fi in enumerate(all_idx):
            frame_boxes=[line[k] for line in per_line_boxes]
            separated=enforce_non_overlapping_lines(
                frame_map[fi],
                frame_boxes,
            )
            for li,box in enumerate(separated):
                per_line_boxes[li][k]=np.asarray(box,dtype=np.float32)
            frame_overlap=0.0
            for li in range(len(separated)-1):
                frame_overlap+=line_overlap_area(
                    separated[li],separated[li+1]
                )
            overlap_pixels+=frame_overlap
            if frame_overlap>0:
                overlap_frames+=1

        subtitle_id=next_subtitle_id
        next_subtitle_id+=1
        for line_id,boxes in enumerate(per_line_boxes):
            track=SubtitleTrack(next_track_id,confirmed=True)
            identity[next_track_id]=(subtitle_id,line_id)
            level=per_line_levels[line_id]
            for k,fi in enumerate(all_idx):
                box=np.asarray(boxes[k],dtype=np.float32)
                track.observations[int(fi)]=TrackObservation(
                    int(fi),box.copy(),None,level,reconstructed=False
                )
                mask_area=_mask_bbox_area(frame_map[fi],box)
                if mask_area is not None:
                    bbox_area=max(
                        1.0,
                        float((box[2]-box[0])*(box[3]-box[1])),
                    )
                    excess_samples.append(
                        max(0.0,(bbox_area-mask_area)/mask_area)
                    )
            tracks.append(track)
            next_track_id+=1

    edge_med,edge_p95=_edge_motion_metrics(tracks)
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
        "line_refinement_applied_count":refinement_applied,
        "line_refinement_anchor_fallback_count":refinement_anchor_fallback,
        "multiline_overlap_pixel_count":int(round(overlap_pixels)),
        "multiline_overlap_frame_count":int(overlap_frames),
        "bbox_excess_area_estimate":(
            float(np.mean(excess_samples)) if excess_samples else 0.0
        ),
        "stabilized_edge_motion_median_px":edge_med,
        "stabilized_edge_motion_p95_px":edge_p95,
    }
    return V55TrackBundle(tracks,identity,metrics)


def _track_center_y(track):
    fs=track.sorted_frames()
    if not fs:
        return 0.0
    vals=[
        .5*(
            float(track.observations[f].bbox[1])
            +float(track.observations[f].bbox[3])
        )
        for f in fs
    ]
    return float(np.median(vals))


def assign_v55_weak_identities(strong_tracks,weak_tracks,identity):
    identity=dict(identity)
    next_subtitle=max((v[0] for v in identity.values()),default=0)+1
    strong_by_id={t.track_id:t for t in strong_tracks}
    for weak in weak_tracks:
        wfs=set(weak.sorted_frames())
        wcy=_track_center_y(weak)
        best=None
        for strong in strong_tracks:
            if strong.track_id not in identity:
                continue
            overlap=len(wfs & set(strong.sorted_frames()))
            if overlap<=0:
                continue
            scy=_track_center_y(strong)
            vertical=abs(wcy-scy)
            key=(overlap,vertical)
            if best is None or key[0]>best[0] or (
                key[0]==best[0] and key[1]<best[1]
            ):
                best=(overlap,vertical,strong)
        if best is None:
            identity[weak.track_id]=(next_subtitle,0)
            next_subtitle+=1
            continue
        sibling=best[2]
        subtitle_id=identity[sibling.track_id][0]
        sibling_line=identity[sibling.track_id][1]
        line_id=sibling_line+1 if wcy>_track_center_y(sibling) else max(0,sibling_line-1)
        identity[weak.track_id]=(subtitle_id,line_id)

    all_tracks=list(strong_tracks)+list(weak_tracks)
    by_sub={}
    for track in all_tracks:
        if track.track_id not in identity:
            continue
        by_sub.setdefault(identity[track.track_id][0],[]).append(track)
    for subtitle_id,group in by_sub.items():
        ordered=sorted(group,key=_track_center_y)
        for line_id,track in enumerate(ordered):
            identity[track.track_id]=(subtitle_id,line_id)
    return identity
def v55_weak_height_baselines(weak_tracks):
    out={}
    for track in weak_tracks:
        heights=[
            max(
                0.0,
                float(track.observations[fi].bbox[3])
                -float(track.observations[fi].bbox[1]),
            )
            for fi in track.sorted_frames()
        ]
        if heights:
            out[int(track.track_id)]=float(np.median(heights))
    return out


def prune_collapsed_v55_weak_tracks(
    weak_tracks,
    baseline_heights,
    min_retained_height_ratio=.45,
):
    kept=[]
    rejected=0
    threshold=float(min_retained_height_ratio)
    for track in weak_tracks:
        heights=[
            max(
                0.0,
                float(track.observations[fi].bbox[3])
                -float(track.observations[fi].bbox[1]),
            )
            for fi in track.sorted_frames()
        ]
        baseline=float(baseline_heights.get(track.track_id,0.0))
        if not heights or baseline<=0.0:
            rejected+=1
            continue
        retained=float(np.median(heights))/baseline
        if retained + 1e-6 < threshold:
            rejected+=1
            continue
        kept.append(track)
    return kept,rejected


def enforce_v55_final_line_separation(video_path,tracks,identity):
    # Resolve cross-lifecycle conflicts before same-subtitle seam clipping.
    # A weak observation has lower authority than a strong observation; if
    # they overlap in area on the same frame but belong to different
    # subtitle lifecycles, suppress the weak observation instead of
    # allowing it to cut into the strong subtitle.
    by_frame={}
    for track in tracks:
        tag=identity.get(track.track_id)
        if tag is None:
            continue
        subtitle_id,line_id=tag
        for fi in track.sorted_frames():
            by_frame.setdefault(int(fi),[]).append(
                (int(subtitle_id),int(line_id),track)
            )

    cross_subtitle_weak_suppressed=0
    for fi,rows in by_frame.items():
        strong_rows=[
            row for row in rows
            if row[2].observations[fi].level!="V5_WEAK"
        ]
        weak_rows=[
            row for row in rows
            if row[2].observations[fi].level=="V5_WEAK"
        ]
        for weak_sid,_,weak_track in weak_rows:
            weak_box=weak_track.observations[fi].bbox
            conflict=any(
                strong_sid!=weak_sid
                and line_overlap_area(
                    weak_box,strong_track.observations[fi].bbox
                )>0
                for strong_sid,_,strong_track in strong_rows
            )
            if conflict:
                weak_track.observations.pop(fi,None)
                cross_subtitle_weak_suppressed+=1

    grouped={}
    for track in tracks:
        tag=identity.get(track.track_id)
        if tag is None:
            continue
        subtitle_id,line_id=tag
        for fi in track.sorted_frames():
            grouped.setdefault((int(fi),int(subtitle_id)),[]).append(
                (int(line_id),track)
            )

    needed=sorted({
        fi for (fi,_),rows in grouped.items()
        if len(rows)>=2
    })
    frame_map=_read_frames(video_path,needed)
    adjustments=0
    overlap_frames=0
    overlap_pixels=0.0

    for (fi,subtitle_id),rows in sorted(grouped.items()):
        if len(rows)<2:
            continue
        rows=sorted(
            rows,
            key=lambda item:(
                item[0],
                _track_center_y(item[1]),
                item[1].track_id,
            ),
        )
        boxes=[
            track.observations[fi].bbox.copy()
            for _,track in rows
        ]
        separated=enforce_non_overlapping_lines(
            frame_map[fi],
            boxes,
        )
        for (_,track),before,after in zip(rows,boxes,separated):
            after=np.asarray(after,dtype=np.float32)
            if not np.allclose(before,after):
                adjustments+=1
                track.observations[fi].bbox=after.copy()

        remaining=0.0
        for a,b in zip(separated,separated[1:]):
            remaining+=line_overlap_area(a,b)
        overlap_pixels+=remaining
        if remaining>0:
            overlap_frames+=1

    return {
        "multiline_overlap_pixel_count":int(round(overlap_pixels)),
        "multiline_overlap_frame_count":int(overlap_frames),
        "final_overlap_pixel_count":int(round(overlap_pixels)),
        "final_overlap_frame_count":int(overlap_frames),
        "final_separation_adjustment_count":int(adjustments),
        "cross_subtitle_weak_suppressed_count":int(
            cross_subtitle_weak_suppressed
        ),
    }

