from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import copy
import json
import math
import time

import cv2
import numpy as np
import torch

from .association import AssociationConfig, build_provisional_tracks
from .benchmark import collect_environment_metadata, latency_summary, real_time_factor
from .chinese_validator import (
    EasyOCRChineseRecognizer,
    representative_frames,
    validate_track,
    validate_weak_track,
)
from .display_shape import (
    build_display_shape_records,
    build_line_shape_records,
    draw_display_polygons,
    draw_line_rectangles,
)
from .fast_backend import FastBackend
from .io import mux_audio, transcode_video, write_coordinate_json
from .io_probe import bottom_roi, probe_video
from .lifecycle import reconstruct_tracks, split_tracks_on_geometry, suppress_reconstructed_overlaps
from .line_grouping import group_candidates_to_lines
from .line_split import split_frame_detections_by_projection
from .horizontal_recovery import recover_tracks_horizontal_extents
from .review import export_event_contact_sheet
from .runtime import configure_runtime, resolve_device, resolve_precision, synchronize
from .smoothing import smooth_tracks, synchronize_tracks
from .types import Candidate, FrameDetections
from .v5_processor import (
    build_v5_strong_tracks,
    discover_v5_weak_tracks,
    learn_global_slot_priors,
)
from .v55_processor import (
    assign_v55_weak_identities,
    build_v55_strong_tracks,
    discover_v55_weak_tracks,
    enforce_v55_final_line_separation,
    prune_collapsed_v55_weak_tracks,
    v55_weak_height_baselines,
)
from .v1_processor import (
    absorb_v1_transition_fragments,
    apply_v1_static_geometry_lock,
    merge_v1_same_content_tracks,
    regularize_v1_single_line_height,
    tighten_v1_static_tracks_with_ocr,
    tighten_v1_static_tracks_with_temporal_glyphs,
)


@dataclass(slots=True)
class PipelineConfig:
    roi_bottom_fraction: float=.45
    high_score: float=.88
    low_score: float=.60
    high_min_area: int=250
    low_min_area: int=30
    max_internal_gap: int=2
    smoothing_window: int=5
    output_codec: str="h264"
    validate_chinese: bool=False
    export_srt: bool=False
    outline_pad_ratio: float=.08
    min_outline_pad: int=3
    temporal_mode: str="v1"
    device: str="auto"
    precision: str="auto"
    batch_size: int=16
    cpu_threads: int=0
    box_thickness: int=2

    def __post_init__(self):
        if not .25 <= self.roi_bottom_fraction <= .45:
            raise ValueError("roi_bottom_fraction must be within 0.25..0.45")
        if self.output_codec not in {"h264","h265"}:
            raise ValueError("output_codec must be h264 or h265")
        if self.max_internal_gap not in {1,2}:
            raise ValueError("max_internal_gap must be 1 or 2")
        if self.smoothing_window<1 or self.smoothing_window%2==0:
            raise ValueError("smoothing_window must be a positive odd integer")
        if self.outline_pad_ratio < 0 or self.min_outline_pad < 0:
            raise ValueError("outline padding must be non-negative")
        if self.temporal_mode not in {"v4","v5","v5_5","v1"}:
            raise ValueError("temporal_mode must be v4, v5, v5_5, or v1")
        if self.device not in {"auto","cpu","cuda"}:
            raise ValueError("device must be auto, cpu, or cuda")
        if self.precision not in {"auto","fp32","fp16"}:
            raise ValueError("precision must be auto, fp32, or fp16")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.cpu_threads < 0:
            raise ValueError("cpu_threads must be non-negative")
        if self.box_thickness <= 0:
            raise ValueError("box_thickness must be positive")


@dataclass(slots=True)
class PipelineResult:
    output_video: Path
    coordinate_json: Path
    review_dir: Path
    cache_path: Path
    metrics: dict


def _cand_to_dict(c):
    return {"bbox":[float(x) for x in c.bbox],"score":float(c.score),
            "level":c.level,"source":c.source}


def _cand_from_dict(d):
    return Candidate(np.array(d["bbox"],np.float32),d["score"],d["level"],d.get("source","cache"))


def _offset(c,y):
    b=c.bbox.copy()
    b[[1,3]]+=float(y)
    return Candidate(b,c.score,c.level,c.source)


def _corner_motion(tracks):
    vals=[]
    for t in tracks:
        fs=t.sorted_frames()
        for a,b in zip(fs,fs[1:]):
            if b-a!=1:
                continue
            d=np.abs(t.observations[a].bbox-t.observations[b].bbox)
            vals.append(float(np.mean(d)))
    if not vals:
        return 0.0,0.0
    return float(np.median(vals)),float(np.percentile(vals,95))


def _default_fast_paths(root: Path):
    repo=Path(root)/"model/FAST"
    checkpoint=repo/"checkpoints/fast_base_ic15_736_finetune_ic17mlt.pth"
    return repo,checkpoint


class SubtitlePipeline:
    def __init__(self,config: PipelineConfig,backend=None,recognizer=None):
        self.config=config
        self.backend=backend
        self.recognizer=recognizer

    def _default_backend(self):
        root=Path(__file__).resolve().parents[2]
        repo,checkpoint=_default_fast_paths(root)
        device=resolve_device(self.config.device)
        precision=resolve_precision(self.config.precision,device)
        configure_runtime(
            device,
            self.config.cpu_threads if self.config.cpu_threads>0 else None,
        )
        return FastBackend(
            repo,
            checkpoint,
            device=device,
            precision=precision,
            batch_size=self.config.batch_size,
            high_score=self.config.high_score,
            low_score=self.config.low_score,
            high_min_area=self.config.high_min_area,
            low_min_area=self.config.low_min_area,
        )

    def _signature(self,source,target):
        st=source.stat()
        return {
            "source":str(source.resolve()),"size":st.st_size,"mtime_ns":st.st_mtime_ns,
            "target_frames":int(target),
            "roi_bottom_fraction":self.config.roi_bottom_fraction,
            "temporal_mode":self.config.temporal_mode,
            "high_score":self.config.high_score,"low_score":self.config.low_score,
            "high_min_area":self.config.high_min_area,"low_min_area":self.config.low_min_area,
            "backend":type(self.backend).__name__ if self.backend is not None else "FastBackend",
        }

    def _cache_path(self,source,output):
        return output.parent/f"{source.stem}.video_text_cache.json"

    def _load_cache(self,path,signature):
        if not path.exists():
            return None
        try:
            d=json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        if d.get("signature")!=signature:
            return None
        out=[]
        for row in d["frames"]:
            out.append(FrameDetections(
                row["frame"],row["timestamp"],
                [_cand_from_dict(x) for x in row["high"]],
                [_cand_from_dict(x) for x in row["low"]],
            ))
        return out

    def _save_cache(self,path,signature,frames):
        rows=[]
        for f in frames:
            rows.append({"frame":f.frame_index,"timestamp":f.timestamp,
                         "high":[_cand_to_dict(x) for x in f.high],
                         "low":[_cand_to_dict(x) for x in f.low]})
        path.write_text(json.dumps({"signature":signature,"frames":rows},
                                   ensure_ascii=False),encoding="utf-8")

    def _detect(self,source,info,target,cache_path):
        signature=self._signature(source,target)
        self._detector_latency_samples_ms=[]
        cached=self._load_cache(cache_path,signature)
        if cached is not None:
            return cached,0.0,True
        if self.backend is None:
            self.backend=self._default_backend()
            signature=self._signature(source,target)
        cap=cv2.VideoCapture(str(source))
        y1,y2=bottom_roi(info.height,self.config.roi_bottom_fraction)
        frames=[]
        processed=0
        batch_size=int(getattr(self.backend,"batch_size",16))
        detector_seconds=0.0
        while processed<target:
            originals=[]; rois=[]
            for _ in range(min(batch_size,target-processed)):
                ok,frame=cap.read()
                if not ok:
                    break
                originals.append(frame)
                rois.append(frame[y1:y2])
            if not originals:
                break
            backend_device=torch.device(
                getattr(self.backend,"device",resolve_device(self.config.device))
            )
            synchronize(backend_device)
            t=time.perf_counter()
            detected=self.backend.detect_batch(rois)
            synchronize(backend_device)
            batch_elapsed=time.perf_counter()-t
            detector_seconds+=batch_elapsed
            if originals:
                per_frame_ms=1000.0*batch_elapsed/len(originals)
                self._detector_latency_samples_ms.extend(
                    [per_frame_ms]*len(originals)
                )
            if len(detected)!=len(originals):
                cap.release()
                raise RuntimeError("detector batch length mismatch")
            for high,low in detected:
                absolute=[_offset(c,y1) for c in list(high)+list(low)]
                grouped=group_candidates_to_lines(absolute)
                gh=[c for c in grouped if c.level=="HIGH"]
                gl=[c for c in grouped if c.level=="LOW"]
                frames.append(FrameDetections(processed,processed/info.fps,gh,gl))
                processed+=1
        cap.release()
        if len(frames)!=target:
            raise RuntimeError(f"decoded {len(frames)} of {target} requested frames")
        self._save_cache(cache_path,signature,frames)
        return frames,detector_seconds,False

    def _load_validation_frames(self,source,tracks):
        needed=set()
        for t in tracks:
            needed.update(representative_frames(t,3))
        if not needed:
            return {}
        cap=cv2.VideoCapture(str(source))
        out={}
        for fi in sorted(needed):
            cap.set(cv2.CAP_PROP_POS_FRAMES,fi)
            ok,frame=cap.read()
            if ok:
                out[fi]=frame
        cap.release()
        return out

    def _validate_language(self,source,tracks):
        counts={"validated_chinese":0,"rejected_non_chinese":0,"retained_uncertain":0}
        if not self.config.validate_chinese:
            return tracks,counts
        try:
            recognizer=self.recognizer or EasyOCRChineseRecognizer(gpu=torch.cuda.is_available())
            frames=self._load_validation_frames(source,tracks)
            kept=[]
            for t in tracks:
                d=validate_track(t,frames,recognizer,max_samples=3)
                counts[d.status]=counts.get(d.status,0)+1
                if d.status!="rejected_non_chinese":
                    kept.append(t)
            return kept,counts
        except Exception:
            for t in tracks:
                t.language_status="retained_uncertain"
            counts["retained_uncertain"]=len(tracks)
            return tracks,counts

    def _validate_v55_weak_post_separation(
        self,source,strong_tracks,weak_tracks,identity,recognizer
    ):
        if not weak_tracks:
            return [],0
        validation_tracks=copy.deepcopy(list(strong_tracks)+list(weak_tracks))
        enforce_v55_final_line_separation(
            source,
            validation_tracks,
            identity,
        )
        weak_ids={t.track_id for t in weak_tracks}
        validation_weak=[
            t for t in validation_tracks if t.track_id in weak_ids
        ]
        frames=self._load_validation_frames(source,validation_weak)
        kept_ids=set()
        rejected=0
        for track in validation_weak:
            decision=validate_weak_track(
                track,frames,recognizer,max_samples=3
            )
            if decision.status=="validated_chinese":
                kept_ids.add(track.track_id)
            else:
                rejected+=1
        kept=[]
        for track in weak_tracks:
            if track.track_id in kept_ids:
                track.language_status="validated_chinese"
                kept.append(track)
        return kept,rejected

    def _records(self,tracks,identity=None):
        rows=[]
        identity=identity or {}
        for t in tracks:
            subtitle_line=identity.get(t.track_id)
            for fi in t.sorted_frames():
                o=t.observations[fi]
                row={
                    "frame":fi,"track_id":t.track_id,
                    "line_id":(
                        int(subtitle_line[1])
                        if subtitle_line is not None else t.track_id
                    ),
                    "bbox":[int(round(x)) for x in o.bbox],
                    "source":o.level,
                    "confidence":None if o.score is None else float(o.score),
                    "reconstructed":bool(o.reconstructed),
                }
                if subtitle_line is not None:
                    row["subtitle_id"]=int(subtitle_line[0])
                rows.append(row)
        return sorted(rows,key=lambda r:(r["frame"],r.get("subtitle_id",r["track_id"]),r["line_id"]))

    def _render(self,source,output,records,info,target):
        by_frame={}
        for r in records:
            by_frame.setdefault(r["frame"],[]).append(r)
        raw=output.with_name(output.stem+".silent_raw.mp4")
        encoded=output.with_name(output.stem+".encoded.mp4")
        cap=cv2.VideoCapture(str(source))
        writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*"mp4v"),info.fps,(info.width,info.height))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"cannot create temporary output: {raw}")
        for fi in range(target):
            ok,frame=cap.read()
            if not ok:
                writer.release(); cap.release()
                raise RuntimeError(f"source ended at frame {fi}")
            if self.config.temporal_mode in {"v5_5","v1"}:
                draw_line_rectangles(
                    frame,
                    by_frame.get(fi,[]),
                    color=(0,255,0),
                    thickness=self.config.box_thickness,
                    line_type=cv2.LINE_AA,
                )
            else:
                draw_display_polygons(
                    frame,
                    by_frame.get(fi,[]),
                    color=(0,255,0),
                    thickness=self.config.box_thickness,
                    line_type=cv2.LINE_AA,
                )
            writer.write(frame)
        writer.release(); cap.release()
        try:
            transcode_video(raw,encoded,self.config.output_codec)
            mux_audio(encoded,source,output)
        finally:
            for p in (raw,encoded):
                try: p.unlink()
                except FileNotFoundError: pass

    def _process_v55(self,source,frames,info,target):
        split_output_count=sum(len(f.high)+len(f.low) for f in frames)
        cfg=AssociationConfig(max_frame_gap=self.config.max_internal_gap+1)
        provisional=build_provisional_tracks(frames,cfg)
        provisional=split_tracks_on_geometry(
            provisional,
            max_internal_gap=self.config.max_internal_gap,
        )
        raw_med,raw_p95=_corner_motion(provisional)

        diagnostic_tracks,events=reconstruct_tracks(
            provisional,
            self.config.max_internal_gap,
            total_frames=target,
        )
        diagnostic_tracks,events,suppressed_reconstructed_overlap_count=(
            suppress_reconstructed_overlaps(diagnostic_tracks,events)
        )

        bundle=build_v55_strong_tracks(
            source,
            frames,
            frame_width=info.width,
            frame_height=info.height,
            max_internal_gap=self.config.max_internal_gap,
        )
        strong_tracks=bundle.tracks
        strong_identity=dict(bundle.identity)
        priors=learn_global_slot_priors(
            strong_tracks,
            frame_width=info.width,
        )
        weak_tracks,weak_metrics=discover_v55_weak_tracks(
            source,
            strong_tracks,
            priors,
            frame_count=target,
            frame_width=info.width,
            frame_height=info.height,
            confirm_frames=3,
        )
        horizontal_recovery_expanded_count=0
        recognizer=self.recognizer

        if self.config.validate_chinese:
            try:
                recognizer=self.recognizer or EasyOCRChineseRecognizer(
                    gpu=torch.cuda.is_available()
                )
                validation_frames=self._load_validation_frames(
                    source,strong_tracks
                )
                kept_strong=[]
                language_counts={
                    "validated_chinese":0,
                    "rejected_non_chinese":0,
                    "retained_uncertain":0,
                }
                pixel_only_rejected=0
                for track in strong_tracks:
                    fs=track.sorted_frames()
                    level=track.observations[fs[0]].level if fs else ""
                    if level=="V55_LINE_PIXEL_ONLY":
                        decision=validate_weak_track(
                            track,validation_frames,recognizer,max_samples=3
                        )
                        if decision.status=="validated_chinese":
                            kept_strong.append(track)
                            language_counts["validated_chinese"]+=1
                        else:
                            pixel_only_rejected+=1
                            language_counts["rejected_non_chinese"]+=1
                    else:
                        decision=validate_track(
                            track,validation_frames,recognizer,max_samples=3
                        )
                        language_counts[decision.status]=(
                            language_counts.get(decision.status,0)+1
                        )
                        if decision.status!="rejected_non_chinese":
                            kept_strong.append(track)

                strong_tracks=kept_strong
                validation_strong_identity={
                    tid:value for tid,value in strong_identity.items()
                    if any(t.track_id==tid for t in strong_tracks)
                }
                validation_identity=assign_v55_weak_identities(
                    strong_tracks,
                    weak_tracks,
                    validation_strong_identity,
                )
                kept_weak,weak_rejected=(
                    self._validate_v55_weak_post_separation(
                        source,
                        strong_tracks,
                        weak_tracks,
                        validation_identity,
                        recognizer,
                    )
                )
                language_counts["validated_chinese"]+=len(kept_weak)
                language_counts["rejected_non_chinese"]+=weak_rejected
                weak_tracks=kept_weak
            except Exception:
                all_tracks=strong_tracks+weak_tracks
                for track in all_tracks:
                    track.language_status="retained_uncertain"
                language_counts={
                    "validated_chinese":0,
                    "rejected_non_chinese":0,
                    "retained_uncertain":len(all_tracks),
                }
                weak_rejected=0
                pixel_only_rejected=0
        else:
            language_counts={
                "validated_chinese":0,
                "rejected_non_chinese":0,
                "retained_uncertain":0,
            }
            weak_rejected=0
            pixel_only_rejected=0

        strong_identity={
            tid:value for tid,value in strong_identity.items()
            if any(t.track_id==tid for t in strong_tracks)
        }

        # Final seam clipping can reveal that a weak line was mostly an
        # overlap with a real strong subtitle. Evaluate that geometry on a
        # deep copy first so a rejected weak line cannot permanently clip a
        # strong line in the real output.
        weak_height_baselines=v55_weak_height_baselines(weak_tracks)
        trial_identity=assign_v55_weak_identities(
            strong_tracks,
            weak_tracks,
            strong_identity,
        )
        trial_tracks=copy.deepcopy(strong_tracks+weak_tracks)
        enforce_v55_final_line_separation(
            source,
            trial_tracks,
            trial_identity,
        )
        weak_ids={t.track_id for t in weak_tracks}
        trial_weak=[
            t for t in trial_tracks if t.track_id in weak_ids
        ]
        kept_trial,weak_geometry_rejected=(
            prune_collapsed_v55_weak_tracks(
                trial_weak,
                weak_height_baselines,
                min_retained_height_ratio=.45,
            )
        )
        kept_weak_ids={t.track_id for t in kept_trial}
        if weak_geometry_rejected:
            weak_tracks=[
                t for t in weak_tracks if t.track_id in kept_weak_ids
            ]
            if self.config.validate_chinese:
                language_counts["validated_chinese"]=max(
                    0,
                    language_counts.get("validated_chinese",0)
                    -weak_geometry_rejected,
                )

        identity=assign_v55_weak_identities(
            strong_tracks,
            weak_tracks,
            strong_identity,
        )
        tracks=strong_tracks+weak_tracks

        v1_merge_metrics={}
        v1_transition_metrics={}
        v1_compact_candidates={}
        if self.config.temporal_mode=="v1":
            (
                tracks,
                identity,
                v1_compact_candidates,
                v1_merge_metrics,
            )=merge_v1_same_content_tracks(
                source,
                tracks,
                identity,
                recognizer=recognizer,
                max_gap=1,
            )
            (
                tracks,
                identity,
                v1_transition_metrics,
            )=absorb_v1_transition_fragments(
                tracks,
                identity,
                max_duration=2,
                min_height_inflation=1.25,
            )

        final_sep_metrics=enforce_v55_final_line_separation(
            source,
            tracks,
            identity,
        )

        v1_metrics={}
        if self.config.temporal_mode=="v1":
            lock_metrics=apply_v1_static_geometry_lock(
                source,
                tracks,
                identity,
                sample_count=15,
                pad_px=2,
                compact_candidates=v1_compact_candidates,
            )
            geometry_recognizer=(
                recognizer
                if recognizer is not None
                and getattr(recognizer,"supports_detection",True)
                else None
            )
            glyph_tighten_metrics=tighten_v1_static_tracks_with_temporal_glyphs(
                source,
                tracks,
                identity,
                geometry_recognizer,
                sample_count=15,
                safety_pad=4,
            )
            if recognizer is not None and getattr(recognizer,"supports_detection",True):
                ocr_tighten_metrics=tighten_v1_static_tracks_with_ocr(
                    source,
                    tracks,
                    identity,
                    recognizer,
                    min_height_px=58,
                    sample_count=3,
                    safety_pad=2,
                )
            else:
                ocr_tighten_metrics={
                    "ocr_tightened_track_count":0,
                    "ocr_tightening_area_ratio_median":1.0,
                }
            height_regularize_metrics=regularize_v1_single_line_height(
                tracks,
                identity,
                frame_height=info.height,
                min_reference_tracks=8,
                center_window_px=16.0,
                safety_extra_px=4.0,
                outlier_extra_px=8.0,
                outlier_ratio=1.15,
            )
            v1_metrics={
                **v1_merge_metrics,
                **v1_transition_metrics,
                **lock_metrics,
                **glyph_tighten_metrics,
                **ocr_tighten_metrics,
                **height_regularize_metrics,
            }

        stab_med,stab_p95=_corner_motion(tracks)
        final_overlap_frame_count=(
            v1_metrics.get(
                "final_overlap_frame_count",
                final_sep_metrics["multiline_overlap_frame_count"],
            )
        )
        final_overlap_pixel_count=(
            v1_metrics.get(
                "final_overlap_pixel_count",
                final_sep_metrics["multiline_overlap_pixel_count"],
            )
        )
        mode_metrics={
            **{f"v55_{k}":v for k,v in bundle.metrics.items()},
            **{f"v55_{k}":v for k,v in weak_metrics.items()},
            **{f"v55_{k}":v for k,v in final_sep_metrics.items()},
            **{f"v1_{k}":v for k,v in v1_metrics.items()},
            "v55_global_slot_prior_count":len(priors),
            "v55_weak_rejected_by_language":weak_rejected,
            "v55_weak_rejected_by_geometry":weak_geometry_rejected,
            "v55_weak_retained_count":len(weak_tracks),
            "v55_pixel_only_rejected_by_language":pixel_only_rejected,
            "multiline_overlap_pixel_count":final_overlap_pixel_count,
            "multiline_overlap_frame_count":final_overlap_frame_count,
            "bbox_excess_area_estimate":bundle.metrics[
                "bbox_excess_area_estimate"
            ],
            "stabilized_edge_motion_median_px":(
                stab_med
                if self.config.temporal_mode=="v1"
                else bundle.metrics["stabilized_edge_motion_median_px"]
            ),
            "stabilized_edge_motion_p95_px":(
                stab_p95
                if self.config.temporal_mode=="v1"
                else bundle.metrics["stabilized_edge_motion_p95_px"]
            ),
            "slot_weak_recovery_count":weak_metrics[
                "slot_weak_recovery_count"
            ],
        }
        return {
            "split_output_count":split_output_count,
            "provisional":provisional,
            "raw_med":raw_med,
            "raw_p95":raw_p95,
            "events":events,
            "suppressed_reconstructed_overlap_count":(
                suppressed_reconstructed_overlap_count
            ),
            "horizontal_recovery_expanded_count":(
                horizontal_recovery_expanded_count
            ),
            "language_counts":language_counts,
            "mode_metrics":mode_metrics,
            "stab_med":stab_med,
            "stab_p95":stab_p95,
            "tracks":tracks,
            "identity":identity,
        }

    def run(self,input_path,output_path,max_frames=0,coordinate_json=None):
        start=time.perf_counter()
        source=Path(input_path).resolve()
        output=Path(output_path).resolve()
        output.parent.mkdir(parents=True,exist_ok=True)
        info=probe_video(source)
        target=min(info.frame_count,int(max_frames)) if max_frames else info.frame_count
        cache_path=self._cache_path(source,output)
        run_device=torch.device(
            getattr(self.backend,"device",resolve_device(self.config.device))
        )
        if run_device.type=="cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(run_device)

        detection_loop_start=time.perf_counter()
        frames,detector_seconds,cache_hit=self._detect(source,info,target,cache_path)
        detection_loop_seconds=time.perf_counter()-detection_loop_start
        temporal_start=time.perf_counter()

        split_input_count=sum(len(f.high)+len(f.low) for f in frames)
        v5_metrics={}
        identity={}

        if self.config.temporal_mode in {"v5_5","v1"}:
            state=self._process_v55(source,frames,info,target)
            split_output_count=state["split_output_count"]
            provisional=state["provisional"]
            raw_med=state["raw_med"]
            raw_p95=state["raw_p95"]
            events=state["events"]
            suppressed_reconstructed_overlap_count=state[
                "suppressed_reconstructed_overlap_count"
            ]
            horizontal_recovery_expanded_count=state[
                "horizontal_recovery_expanded_count"
            ]
            language_counts=state["language_counts"]
            v5_metrics=state["mode_metrics"]
            stab_med=state["stab_med"]
            stab_p95=state["stab_p95"]
            tracks=state["tracks"]
            identity=state["identity"]

        elif self.config.temporal_mode=="v5":
            # V5 deliberately does NOT mutate detections frame-by-frame.
            # FAST is used only for coarse temporal segmentation. Vertical
            # line slots and horizontal extents are inferred by multi-frame
            # consensus for each subtitle segment.
            split_output_count=split_input_count
            cfg=AssociationConfig(max_frame_gap=self.config.max_internal_gap+1)
            provisional=build_provisional_tracks(frames,cfg)
            provisional=split_tracks_on_geometry(
                provisional,
                max_internal_gap=self.config.max_internal_gap,
            )
            raw_med,raw_p95=_corner_motion(provisional)

            # Keep lifecycle events only for diagnostics/review. They no
            # longer define final geometry in V5.
            diagnostic_tracks,events=reconstruct_tracks(
                provisional,
                self.config.max_internal_gap,
                total_frames=target,
            )
            diagnostic_tracks,events,suppressed_reconstructed_overlap_count=(
                suppress_reconstructed_overlaps(diagnostic_tracks,events)
            )

            strong_tracks,strong_metrics=build_v5_strong_tracks(
                source,
                frames,
                frame_width=info.width,
                frame_height=info.height,
                max_internal_gap=self.config.max_internal_gap,
            )
            priors=learn_global_slot_priors(
                strong_tracks,
                frame_width=info.width,
            )
            weak_tracks,weak_metrics=discover_v5_weak_tracks(
                source,
                strong_tracks,
                priors,
                frame_count=target,
                frame_width=info.width,
                frame_height=info.height,
                confirm_frames=3,
            )
            horizontal_recovery_expanded_count=0

            if self.config.validate_chinese:
                try:
                    recognizer=self.recognizer or EasyOCRChineseRecognizer(
                        gpu=torch.cuda.is_available()
                    )
                    validation_frames=self._load_validation_frames(
                        source,strong_tracks+weak_tracks
                    )
                    kept_strong=[]
                    language_counts={
                        "validated_chinese":0,
                        "rejected_non_chinese":0,
                        "retained_uncertain":0,
                    }
                    pixel_only_rejected=0
                    for track in strong_tracks:
                        fs=track.sorted_frames()
                        level=(
                            track.observations[fs[0]].level
                            if fs else ""
                        )
                        if level=="V5_SLOT_PIXEL_ONLY":
                            decision=validate_weak_track(
                                track,validation_frames,recognizer,max_samples=3
                            )
                            if decision.status=="validated_chinese":
                                kept_strong.append(track)
                                language_counts["validated_chinese"]+=1
                            else:
                                pixel_only_rejected+=1
                                language_counts["rejected_non_chinese"]+=1
                        else:
                            decision=validate_track(
                                track,validation_frames,recognizer,max_samples=3
                            )
                            language_counts[decision.status]=(
                                language_counts.get(decision.status,0)+1
                            )
                            if decision.status!="rejected_non_chinese":
                                kept_strong.append(track)

                    kept_weak=[]
                    weak_rejected=0
                    for track in weak_tracks:
                        decision=validate_weak_track(
                            track,validation_frames,recognizer,max_samples=3
                        )
                        if decision.status=="validated_chinese":
                            kept_weak.append(track)
                            language_counts["validated_chinese"]+=1
                        else:
                            weak_rejected+=1
                            language_counts["rejected_non_chinese"]+=1
                    strong_tracks=kept_strong
                    weak_tracks=kept_weak
                except Exception:
                    # Preserve recall if the optional recognizer is unavailable.
                    tracks=strong_tracks+weak_tracks
                    for track in tracks:
                        track.language_status="retained_uncertain"
                    language_counts={
                        "validated_chinese":0,
                        "rejected_non_chinese":0,
                        "retained_uncertain":len(tracks),
                    }
                    weak_rejected=0
                    pixel_only_rejected=0
            else:
                language_counts={
                    "validated_chinese":0,
                    "rejected_non_chinese":0,
                    "retained_uncertain":0,
                }
                weak_rejected=0
                pixel_only_rejected=0

            tracks=strong_tracks+weak_tracks
            v5_metrics={
                **{f"v5_{k}":v for k,v in strong_metrics.items()},
                **{f"v5_{k}":v for k,v in weak_metrics.items()},
                "v5_global_slot_prior_count":len(priors),
                "v5_weak_rejected_by_language":weak_rejected,
                "v5_weak_retained_count":len(weak_tracks),
                "v5_pixel_only_rejected_by_language":pixel_only_rejected,
            }
            stab_med,stab_p95=_corner_motion(tracks)
        else:
            # Legacy V4 path retained for A/B regression comparison.
            split_cap=cv2.VideoCapture(str(source))
            split_frames=[]
            for fd in frames:
                ok,frame=split_cap.read()
                if not ok:
                    split_cap.release()
                    raise RuntimeError(f"cannot decode frame {fd.frame_index} for line splitting")
                split_frames.append(split_frame_detections_by_projection(frame,fd))
            split_cap.release()
            frames=split_frames
            split_output_count=sum(len(f.high)+len(f.low) for f in frames)

            cfg=AssociationConfig(max_frame_gap=self.config.max_internal_gap+1)
            provisional=build_provisional_tracks(frames,cfg)
            provisional=split_tracks_on_geometry(
                provisional,
                max_internal_gap=self.config.max_internal_gap,
            )
            raw_med,raw_p95=_corner_motion(provisional)
            tracks,events=reconstruct_tracks(
                provisional,
                self.config.max_internal_gap,
                total_frames=target,
            )
            tracks,events,suppressed_reconstructed_overlap_count=(
                suppress_reconstructed_overlaps(tracks,events)
            )
            tracks,horizontal_recovery_expanded_count=recover_tracks_horizontal_extents(
                source,
                tracks,
            )
            tracks,language_counts=self._validate_language(source,tracks)
            tracks=smooth_tracks(tracks,self.config.smoothing_window)
            tracks=synchronize_tracks(
                tracks,
                frame_width=info.width,
                frame_height=info.height,
                outline_pad_ratio=self.config.outline_pad_ratio,
                min_outline_pad=self.config.min_outline_pad,
            )
            stab_med,stab_p95=_corner_motion(tracks)
        records=self._records(tracks,identity=identity)
        for r in records:
            r["timestamp"]=r["frame"]/info.fps
        if self.config.temporal_mode in {"v5_5","v1"}:
            display_shapes=build_line_shape_records(records,info.fps)
        else:
            display_shapes=build_display_shape_records(records,info.fps)
        temporal_postprocess_seconds=time.perf_counter()-temporal_start
        render_start=time.perf_counter()
        self._render(source,output,records,info,target)
        render_encode_seconds=time.perf_counter()-render_start
        production_elapsed_seconds=time.perf_counter()-start

        review_dir=output.with_name(output.stem+"_review")
        review_dir.mkdir(parents=True,exist_ok=True)
        for event in events:
            if event.event_type in {"internal_miss_recovered","track_end","track_start_backfill"}:
                export_event_contact_sheet(source,event,review_dir,self.config.roi_bottom_fraction,context=2)

        gaps=[e for e in events if e.event_type=="internal_miss_recovered"]
        used_low=sum(1 for t in provisional for o in t.observations.values() if o.level=="LOW")
        total_low=sum(len(f.low) for f in frames)
        actual_device=torch.device(
            getattr(self.backend,"device",run_device)
        )
        actual_precision=str(
            getattr(
                self.backend,
                "precision",
                resolve_precision(self.config.precision,actual_device),
            )
        )
        actual_batch_size=int(
            getattr(self.backend,"batch_size",self.config.batch_size)
        )
        actual_cpu_threads=(
            int(torch.get_num_threads()) if actual_device.type=="cpu" else 0
        )
        latency=latency_summary(
            getattr(self,"_detector_latency_samples_ms",[])
        )
        source_duration_seconds=target/max(info.fps,1e-9)
        metrics={
            "frames":target,"source_fps":info.fps,"width":info.width,"height":info.height,
            "source_codec":info.fourcc,
            "source_duration_seconds":source_duration_seconds,
            "roi_bottom_fraction":self.config.roi_bottom_fraction,
            "temporal_mode":self.config.temporal_mode,
            "device":actual_device.type,
            "precision":actual_precision,
            "batch_size":actual_batch_size,
            "cpu_threads":actual_cpu_threads,
            "detection_loop_seconds":detection_loop_seconds,
            "detector_seconds":detector_seconds,
            "detector_fps":None if detector_seconds<=0 else target/detector_seconds,
            "detector_latency_mean_ms":latency["mean_ms"],
            "detector_latency_p50_ms":latency["p50_ms"],
            "detector_latency_p95_ms":latency["p95_ms"],
            "temporal_postprocess_seconds":temporal_postprocess_seconds,
            "render_encode_seconds":render_encode_seconds,
            "cache_hit":cache_hit,
            "peak_vram_mb":(
                float(torch.cuda.max_memory_allocated(run_device)/1024**2)
                if run_device.type=="cuda" else 0.0
            ),
            "high_candidate_count":sum(len(f.high) for f in frames),
            "low_candidate_count":total_low,
            "line_split_input_candidate_count":split_input_count,
            "line_split_output_candidate_count":split_output_count,
            "horizontal_recovery_expanded_count":horizontal_recovery_expanded_count,
            "track_count":len(tracks),
            "track_end_count":sum(e.event_type=="track_end" for e in events),
            "one_frame_internal_gap_count":sum((e.end_frame-e.start_frame+1)==1 for e in gaps),
            "two_frame_internal_gap_count":sum((e.end_frame-e.start_frame+1)==2 for e in gaps),
            "internal_miss_recovered_frames":sum(e.end_frame-e.start_frame+1 for e in gaps),
            "suppressed_reconstructed_overlap_count":suppressed_reconstructed_overlap_count,
            "start_backfill_frames":sum(e.end_frame-e.start_frame+1 for e in events if e.event_type=="track_start_backfill"),
            "weak_rejected_count":max(0,total_low-used_low),
            "unresolved_gap_frames":0,
            "temporal_extension_frames":0,
            "boxed_frames":len({r["frame"] for r in records}),
            "display_shape_count":len(display_shapes),
            "multiline_polygon_frames":len({
                row["frame"] for row in display_shapes if len(row["track_ids"])>1
            }),
            "raw_box_corner_motion_median_px":raw_med,
            "raw_box_corner_motion_p95_px":raw_p95,
            "stabilized_corner_motion_median_px":stab_med,
            "stabilized_corner_motion_p95_px":stab_p95,
            **language_counts,
            **v5_metrics,
        }
        metrics["elapsed_seconds"]=production_elapsed_seconds
        metrics["end_to_end_fps"]=target/max(production_elapsed_seconds,1e-9)
        metrics["real_time_factor"]=real_time_factor(
            production_elapsed_seconds,source_duration_seconds
        )
        metrics["output_frame_count"]=target
        metrics["dropped_frame_count"]=0
        root=Path(__file__).resolve().parents[2]
        _,checkpoint=_default_fast_paths(root)
        metrics["environment"]=collect_environment_metadata(root,checkpoint)

        coordinate_path=Path(coordinate_json).resolve() if coordinate_json else output.with_suffix(".json")
        metadata={"input":str(source),"output":str(output),**metrics}
        write_coordinate_json(
            coordinate_path,
            metadata,
            records,
            events,
            display_shapes=display_shapes,
            frame_count=target,
            fps=info.fps,
        )
        return PipelineResult(output,coordinate_path,review_dir,cache_path,metrics)
