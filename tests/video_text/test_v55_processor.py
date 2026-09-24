from pathlib import Path

import cv2
import numpy as np

from src.video_text.types import SubtitleTrack, TrackObservation
from src.video_text.v5_weak_birth import SlotPrior
from src.video_text.v55_processor import discover_v55_weak_tracks


ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/"tests/video_text/.tmp_v55_processor"


def _outlined_rect(frame,x1,y1,x2,y2):
    cv2.rectangle(frame,(x1-3,y1-3),(x2+3,y2+3),(10,10,10),-1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(245,245,245),-1)


def _strong_track(track_id,start,end,box):
    t=SubtitleTrack(track_id,confirmed=True)
    for f in range(start,end+1):
        t.observations[f]=TrackObservation(
            f,np.array(box,np.float32),None,"V5_SLOT"
        )
    return t


def _make_two_slot_video(path,n=10,bottom_range=range(3,9)):
    path.parent.mkdir(parents=True,exist_ok=True)
    w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,220))
    assert w.isOpened()
    for i in range(n):
        f=np.full((220,320,3),80,np.uint8)
        for x in (95,135,175,215):
            _outlined_rect(f,x,82,x+22,118)
        if i in bottom_range:
            _outlined_rect(f,145,150,175,185)
        w.write(f)
    w.release()


def test_bottom_weak_line_can_be_recovered_while_top_strong_line_covers_same_frames():
    src=WORK/"top_strong_bottom_weak.mp4"
    _make_two_slot_video(src)
    strong=[_strong_track(1,0,9,[88,76,246,125])]
    priors=[
        SlotPrior(74,128,167,54,expected_x1=88,expected_x2=246),
        SlotPrior(142,193,160,51),
    ]

    weak,metrics=discover_v55_weak_tracks(
        src,strong,priors,frame_count=10,
        frame_width=320,frame_height=220,confirm_frames=3,
    )
    assert len(weak)==1
    assert weak[0].sorted_frames()==list(range(3,9))
    b=weak[0].observations[3].bbox
    assert 155 <= (b[1]+b[3])*.5 <= 180
    assert set(weak[0].sorted_frames()) & set(strong[0].sorted_frames())
    assert metrics["slot_weak_recovery_count"]==1


def test_same_slot_strong_coverage_suppresses_duplicate_weak_birth():
    src=WORK/"same_slot_suppressed.mp4"
    _make_two_slot_video(src)
    strong=[_strong_track(1,3,8,[138,143,182,193])]
    priors=[SlotPrior(142,193,160,51)]

    weak,metrics=discover_v55_weak_tracks(
        src,strong,priors,frame_count=10,
        frame_width=320,frame_height=220,confirm_frames=3,
    )

    assert weak==[]
    assert metrics["weak_track_count"]==0
    assert metrics["slot_weak_recovery_count"]==0


def _make_detections(n,boxes,fps=10.0):
    from src.video_text.types import Candidate, FrameDetections
    rows=[]
    for i in range(n):
        high=[
            Candidate(np.array(box,np.float32),.96,"HIGH","fake")
            for box in boxes
        ]
        rows.append(FrameDetections(i,i/fps,high,[]))
    return rows


def test_v55_two_line_identity_and_final_boxes_never_overlap():
    from src.video_text.v55_processor import build_v55_strong_tracks
    src=WORK/"strong_two_lines.mp4"
    src.parent.mkdir(parents=True,exist_ok=True)
    w=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,220))
    assert w.isOpened()
    for _ in range(10):
        f=np.full((220,320,3),100,np.uint8)
        for x in (65,105,145,185,225):
            _outlined_rect(f,x,82,x+22,115)
        _outlined_rect(f,145,123,175,157)
        w.write(f)
    w.release()
    dets=_make_detections(
        10,
        [
            [58,76,270,130],
            [138,112,182,163],
        ],
    )
    bundle=build_v55_strong_tracks(
        src,dets,frame_width=320,frame_height=220,
        max_internal_gap=2,bootstrap_count=7,sample_count=7,
    )

    assert len(bundle.tracks)==2
    identities=[bundle.identity[t.track_id] for t in bundle.tracks]
    assert {x[0] for x in identities}=={1}
    assert {x[1] for x in identities}=={0,1}
    ordered=sorted(
        bundle.tracks,
        key=lambda t:bundle.identity[t.track_id][1],
    )
    for fi in range(10):
        top=ordered[0].observations[fi].bbox
        bottom=ordered[1].observations[fi].bbox
        assert top[3] <= bottom[1]
    assert bundle.metrics["multiline_overlap_frame_count"]==0


def test_v55_tighter_than_v54_without_clipping_known_outline():
    from src.video_text.v5_processor import build_v5_strong_tracks
    from src.video_text.v55_processor import build_v55_strong_tracks
    src=WORK/"tightness_compare.mp4"
    w=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(360,220))
    assert w.isOpened()
    for _ in range(10):
        f=np.full((220,360,3),100,np.uint8)
        for x in (90,130,170,210,250):
            _outlined_rect(f,x,92,x+22,128)
        w.write(f)
    w.release()
    dets=_make_detections(10,[[60,78,300,144]])
    v54,_=build_v5_strong_tracks(
        src,dets,frame_width=360,frame_height=220,
        bootstrap_count=7,sample_count=7,
    )
    v55=build_v55_strong_tracks(
        src,dets,frame_width=360,frame_height=220,
        bootstrap_count=7,sample_count=7,
    )
    b54=v54[0].observations[0].bbox
    b55=v55.tracks[0].observations[0].bbox
    area54=(b54[2]-b54[0])*(b54[3]-b54[1])
    area55=(b55[2]-b55[0])*(b55[3]-b55[1])
    assert area55 < area54
    assert b55[0] <= 87 and b55[2] >= 275
    assert b55[1] <= 89 and b55[3] >= 131


def test_v55_tracks_slow_horizontal_motion_with_bounded_edge_steps():
    from src.video_text.v55_processor import build_v55_strong_tracks
    src=WORK/"slow_motion.mp4"
    w=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(360,220))
    assert w.isOpened()
    for i in range(10):
        f=np.full((220,360,3),100,np.uint8)
        dx=i
        for x in (100,140,180,220):
            _outlined_rect(f,x+dx,92,x+22+dx,128)
        w.write(f)
    w.release()
    dets=_make_detections(10,[[80,80,290,142]])
    bundle=build_v55_strong_tracks(
        src,dets,frame_width=360,frame_height=220,
        bootstrap_count=7,sample_count=7,refinement_window=3,
    )
    track=bundle.tracks[0]
    xs=[float(track.observations[i].bbox[0]) for i in range(10)]
    assert xs[-1] > xs[0]
    assert max(abs(b-a) for a,b in zip(xs,xs[1:])) <= 2.01
    assert bundle.metrics["stabilized_edge_motion_p95_px"] <= 2.01


def test_final_line_separation_includes_weak_sibling_tracks():
    from src.video_text.v55_processor import enforce_v55_final_line_separation
    src=WORK/"final_weak_overlap.mp4"
    w=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,220))
    assert w.isOpened()
    f=np.full((220,320,3),100,np.uint8)
    _outlined_rect(f,80,82,250,118)
    _outlined_rect(f,135,130,185,166)
    w.write(f)
    w.release()

    top=_strong_track(1,0,0,[76,78,254,137])
    bottom=_strong_track(2,0,0,[130,116,190,172])
    bottom.observations[0].level="V5_WEAK"
    identity={1:(7,0),2:(7,1)}

    metrics=enforce_v55_final_line_separation(
        src,[top,bottom],identity
    )
    a=top.observations[0].bbox
    b=bottom.observations[0].bbox
    assert a[3] <= b[1]
    assert a[0] == 76 and a[2] == 254
    assert b[0] == 130 and b[2] == 190
    assert metrics["final_overlap_frame_count"]==0
    assert metrics["final_separation_adjustment_count"]>=1


def test_cross_subtitle_weak_overlap_is_suppressed_at_boundary():
    from src.video_text.v55_processor import enforce_v55_final_line_separation
    src=WORK/"cross_subtitle_boundary.mp4"
    w=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,220))
    assert w.isOpened()
    f=np.full((220,320,3),100,np.uint8)
    _outlined_rect(f,80,82,250,118)
    _outlined_rect(f,135,130,185,166)
    w.write(f)
    w.release()

    strong=_strong_track(1,0,0,[76,78,254,137])
    weak=_strong_track(2,0,0,[130,116,190,172])
    weak.observations[0].level="V5_WEAK"
    identity={1:(8,0),2:(7,1)}

    metrics=enforce_v55_final_line_separation(
        src,[strong,weak],identity
    )
    assert strong.sorted_frames()==[0]
    assert weak.sorted_frames()==[]
    assert metrics["cross_subtitle_weak_suppressed_count"]==1
    assert metrics["final_overlap_frame_count"]==0


def test_post_separation_prunes_collapsed_weak_line_but_keeps_intact_one():
    from src.video_text.v55_processor import prune_collapsed_v55_weak_tracks

    collapsed=_strong_track(201,0,2,[350,898,415,913])
    collapsed.observations[0].level="V5_WEAK"
    collapsed.observations[1].level="V5_WEAK"
    collapsed.observations[2].level="V5_WEAK"

    intact=_strong_track(202,10,12,[330,846,390,906])
    intact.observations[10].level="V5_WEAK"
    intact.observations[11].level="V5_WEAK"
    intact.observations[12].level="V5_WEAK"

    baseline_heights={201:60.0,202:60.0}
    kept,rejected=prune_collapsed_v55_weak_tracks(
        [collapsed,intact],
        baseline_heights,
        min_retained_height_ratio=.45,
    )
    assert [t.track_id for t in kept]==[202]
    assert rejected==1


def test_weak_identity_tie_prefers_nearest_vertical_strong_track():
    from src.video_text.v55_processor import assign_v55_weak_identities

    top=_strong_track(301,0,4,[60,50,260,90])
    bottom=_strong_track(302,0,4,[90,150,230,190])
    weak=_strong_track(303,0,4,[135,95,175,125])
    for fi in weak.sorted_frames():
        weak.observations[fi].level="V5_WEAK"

    identity=assign_v55_weak_identities(
        [top,bottom],
        [weak],
        {301:(10,0),302:(11,0)},
    )

    assert identity[303][0]==10


def test_weak_birth_rejects_persistent_glyph_far_from_slot_center():
    src=WORK/"weak_offcenter_noise.mp4"
    src.parent.mkdir(parents=True,exist_ok=True)
    w=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,220))
    assert w.isOpened()
    for _ in range(6):
        f=np.full((220,320,3),80,np.uint8)
        # Inside the old weak-birth tolerance (~0.8 font heights) but far
        # enough from the learned subtitle center to be scene/UI noise.
        _outlined_rect(f,185,150,209,185)
        w.write(f)
    w.release()

    weak,metrics=discover_v55_weak_tracks(
        src,[],
        [SlotPrior(142,193,160,51)],
        frame_count=6,frame_width=320,frame_height=220,
        confirm_frames=3,
    )

    assert weak==[]
    assert metrics["weak_track_count"]==0


def test_pixel_only_filter_keeps_prior_aligned_and_rejects_bad_geometry():
    from src.video_text.v55_processor import filter_v55_pixel_only_tracks

    def pixel(track_id,box):
        t=_strong_track(track_id,0,4,box)
        for fi in t.sorted_frames():
            t.observations[fi].level="V55_LINE_PIXEL_ONLY"
        return t

    good=pixel(404,[137,149,183,190])
    off_center=pixel(405,[240,149,286,190])
    oversized=pixel(406,[100,120,220,210])

    kept,rejected=filter_v55_pixel_only_tracks(
        [good,off_center,oversized],
        [SlotPrior(142,193,160,51)],
    )

    assert [t.track_id for t in kept]==[404]
    assert rejected==2
