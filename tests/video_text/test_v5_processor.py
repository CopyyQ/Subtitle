from pathlib import Path

import cv2
import numpy as np

from src.video_text.types import Candidate, FrameDetections
from src.video_text.v5_processor import build_v5_strong_tracks


ROOT=Path(__file__).resolve().parents[2]
WORK=ROOT/"tests/video_text/.tmp_v5_processor"


def _outlined_rect(frame,x1,y1,x2,y2):
    cv2.rectangle(frame,(x1-3,y1-3),(x2+3,y2+3),(10,10,10),-1)
    cv2.rectangle(frame,(x1,y1),(x2,y2),(245,245,245),-1)


def _make_video(path,n=10,two_lines=True,noise_frame=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,200))
    assert w.isOpened()
    for i in range(n):
        f=np.full((200,320,3),100,np.uint8)
        for x in range(65,246,36):
            _outlined_rect(f,x,95,x+22,127)
        if two_lines:
            _outlined_rect(f,145,140,175,170)
        if noise_frame is not None and i==noise_frame:
            _outlined_rect(f,8,94,40,128)
        w.write(f)
    w.release()


def _raw_frames(n=10):
    out=[]
    for i in range(n):
        c=Candidate(np.array([58,88,270,132],np.float32),.96,"HIGH","fake")
        out.append(FrameDetections(i,i/10.0,[c],[]))
    return out


def test_v5_strong_segment_locks_two_line_slots_and_fills_segment():
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"two_line.mp4"
    _make_video(src,two_lines=True)

    tracks,metrics=build_v5_strong_tracks(
        src,_raw_frames(),frame_width=320,frame_height=200,
        max_internal_gap=2,bootstrap_count=7,sample_count=7,
    )

    assert len(tracks)==2
    assert metrics["segment_count"]==1
    assert metrics["two_line_segment_count"]==1
    assert [t.sorted_frames() for t in tracks]==[list(range(10)),list(range(10))]
    top,bottom=sorted(
        tracks,
        key=lambda t: t.observations[0].bbox[1],
    )
    assert top.observations[0].bbox[3] < bottom.observations[0].bbox[3]
    assert len({tuple(t.observations[f].bbox) for t in tracks for f in t.sorted_frames()})==2


def test_v5_temporal_consensus_does_not_stretch_track_from_one_noise_frame():
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"noise.mp4"
    _make_video(src,two_lines=False,noise_frame=3)

    tracks,metrics=build_v5_strong_tracks(
        src,_raw_frames(),frame_width=320,frame_height=200,
        max_internal_gap=2,bootstrap_count=7,sample_count=7,
    )

    assert len(tracks)==1
    box=tracks[0].observations[0].bbox
    assert box[0] > 45
    assert box[2] < 285
    assert all(
        np.allclose(tracks[0].observations[f].bbox,box)
        for f in tracks[0].sorted_frames()
    )


def _strong_track(track_id,start,end,box):
    from src.video_text.types import SubtitleTrack, TrackObservation
    t=SubtitleTrack(track_id,confirmed=True)
    for f in range(start,end+1):
        t.observations[f]=TrackObservation(
            f,np.array(box,np.float32),None,"V5_SLOT"
        )
    return t


def test_v5_discovers_persistent_single_character_inside_uncovered_gap():
    from src.video_text.v5_processor import (
        discover_v5_weak_tracks,
        learn_global_slot_priors,
    )

    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"weak_gap.mp4"
    w=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,200))
    assert w.isOpened()
    for i in range(30):
        f=np.full((200,320,3),80,np.uint8)
        if 10<=i<=15:
            _outlined_rect(f,145,95,175,130)
        w.write(f)
    w.release()

    strong=[
        _strong_track(1,0,4,[80,86,240,138]),
        _strong_track(2,20,29,[90,86,230,138]),
    ]
    priors=learn_global_slot_priors(strong,frame_width=320)
    weak,metrics=discover_v5_weak_tracks(
        src,strong,priors,frame_count=30,
        frame_width=320,frame_height=200,
        confirm_frames=3,
    )

    assert len(weak)==1
    assert weak[0].sorted_frames()==list(range(10,16))
    assert metrics["weak_track_count"]==1
    assert metrics["weak_frame_count"]==6


def test_v5_weak_discovery_does_not_box_blank_gap():
    from src.video_text.v5_processor import (
        discover_v5_weak_tracks,
        learn_global_slot_priors,
    )

    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"blank_gap.mp4"
    w=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,200))
    assert w.isOpened()
    for _ in range(20):
        w.write(np.full((200,320,3),80,np.uint8))
    w.release()

    strong=[
        _strong_track(1,0,4,[80,86,240,138]),
        _strong_track(2,15,19,[90,86,230,138]),
    ]
    priors=learn_global_slot_priors(strong,frame_width=320)
    weak,metrics=discover_v5_weak_tracks(
        src,strong,priors,frame_count=20,
        frame_width=320,frame_height=200,
        confirm_frames=3,
    )

    assert weak==[]
    assert metrics["weak_track_count"]==0
    assert metrics["weak_frame_count"]==0


def test_concurrent_coarse_rows_are_used_as_two_layout_slots():
    from src.video_text.v5_processor import coarse_line_slots_from_tracks
    from src.video_text.v5_segments import build_subtitle_segments

    top=_strong_track(10,0,9,[82,820,640,879])
    bottom=_strong_track(11,0,9,[333,875,386,926])
    segment=build_subtitle_segments([top,bottom])[0]

    slots=coarse_line_slots_from_tracks(
        segment,{10:top,11:bottom},frame_width=720,frame_height=1280
    )

    assert len(slots)==2
    assert slots[0].y1 <= 817 and slots[0].y2 >= 882
    assert slots[1].y1 <= 872 and slots[1].y2 >= 929


def test_sequential_same_lane_fragments_do_not_fake_two_line_layout():
    from src.video_text.v5_processor import coarse_line_slots_from_tracks
    from src.video_text.v5_segments import build_subtitle_segments

    a=_strong_track(10,0,9,[170,849,548,903])
    b=_strong_track(11,10,19,[209,849,509,903])
    # They may be one or two temporal segments depending on content-change
    # splitting; either way a non-overlapping pair must never mean two rows.
    class Segment:
        start_frame=0
        end_frame=19
        track_ids=[10,11]
    slots=coarse_line_slots_from_tracks(
        Segment(),{10:a,11:b},frame_width=720,frame_height=1280
    )

    assert len(slots)==0


def test_strong_processor_prefers_concurrent_coarse_rows_over_merged_pixel_bootstrap():
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"coarse_two_rows.mp4"
    w=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,220))
    assert w.isOpened()
    dets=[]
    for i in range(10):
        f=np.full((220,320,3),100,np.uint8)
        for x in range(65,246,36):
            _outlined_rect(f,x,82,x+22,112)
        _outlined_rect(f,145,122,175,152)
        # Stable outlined bridge deliberately makes image-row projection look
        # like one tall block. Coarse FAST rows must still win.
        _outlined_rect(f,154,108,166,126)
        w.write(f)
        dets.append(FrameDetections(
            i,i/10.0,
            [
                Candidate(np.array([58,76,270,116],np.float32),.96,"HIGH","fake"),
                Candidate(np.array([138,118,182,156],np.float32),.94,"HIGH","fake"),
            ],
            [],
        ))
    w.release()

    tracks,metrics=build_v5_strong_tracks(
        src,dets,frame_width=320,frame_height=220,
        max_internal_gap=2,bootstrap_count=7,sample_count=7,
    )

    assert len(tracks)==2
    boxes=sorted(
        [t.observations[0].bbox for t in tracks],
        key=lambda b:b[1],
    )
    assert boxes[0][3] < boxes[1][3]
    assert metrics["two_line_segment_count"]==1


def test_two_line_analysis_crops_do_not_let_top_row_expand_lower_x():
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"overlapping_y_rows.mp4"
    w=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(720,1280))
    assert w.isOpened()
    dets=[]
    for i in range(8):
        f=np.full((1280,720,3),90,np.uint8)
        # Wide top line, close enough in Y that padded detector boxes overlap.
        for x in range(85,630,42):
            _outlined_rect(f,x,824,x+27,874)
        # Short centered lower line.
        _outlined_rect(f,340,878,380,923)
        w.write(f)
        dets.append(FrameDetections(
            i,i/10.0,
            [
                Candidate(np.array([82,819,640,879],np.float32),.96,"HIGH","fake"),
                Candidate(np.array([333,875,386,926],np.float32),.94,"HIGH","fake"),
            ],
            [],
        ))
    w.release()

    tracks,_=build_v5_strong_tracks(
        src,dets,frame_width=720,frame_height=1280,
        max_internal_gap=2,bootstrap_count=7,sample_count=7,
    )
    assert len(tracks)==2
    bottom=max(tracks,key=lambda t:t.observations[0].bbox[1])
    b=bottom.observations[0].bbox

    # The lower line is one compact glyph. Pixels from the wide top line must
    # not leak into its horizontal consensus.
    assert b[2]-b[0] < 90


def test_single_line_output_y_uses_coarse_row_not_background_projection():
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"single_line_y_anchor.mp4"
    w=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(320,220))
    assert w.isOpened()
    dets=[]
    for i in range(8):
        f=np.full((220,320,3),100,np.uint8)
        # Persistent outlined background detail above the subtitle that can
        # pollute bootstrap projection.
        _outlined_rect(f,150,55,166,72)
        for x in range(85,236,36):
            _outlined_rect(f,x,100,x+22,135)
        w.write(f)
        dets.append(FrameDetections(
            i,i/10.0,
            [Candidate(np.array([80,96,250,139],np.float32),.96,"HIGH","fake")],
            [],
        ))
    w.release()

    tracks,_=build_v5_strong_tracks(
        src,dets,frame_width=320,frame_height=220,
        max_internal_gap=2,bootstrap_count=7,sample_count=7,
    )
    assert len(tracks)==1
    b=tracks[0].observations[0].bbox
    # Detector row is y=96..139; outline padding is fine, but the unrelated
    # feature at y=55 must not pull the locked slot upward.
    assert b[1] >= 90
    assert b[3] <= 145


def test_single_coarse_row_creates_primary_locked_y_slot():
    from src.video_text.v5_processor import coarse_supported_slots_from_tracks
    from src.video_text.v5_segments import build_subtitle_segments

    row=_strong_track(10,0,20,[165,849,557,903])
    segment=build_subtitle_segments([row])[0]
    slots=coarse_supported_slots_from_tracks(
        segment,{10:row},frame_width=720,frame_height=1280
    )

    assert len(slots)==1
    slot=slots[0]
    assert 843 <= slot.y1 <= 846
    assert 906 <= slot.y2 <= 909


def test_pixel_bootstrap_only_adds_vertical_lane_not_already_supported_by_fast():
    from src.video_text.v5_processor import merge_layout_slots
    from src.video_text.v5_weak_birth import SlotPrior

    coarse=[SlotPrior(844,908,360,64)]
    pixel=[
        SlotPrior(825,910,360,85),  # overlaps coarse main row: reject
        SlotPrior(879,927,360,48),  # distinct lower row: keep
    ]
    merged=merge_layout_slots(coarse,pixel,max_slots=2)

    assert len(merged)==2
    assert merged[0][1] is True
    assert merged[1][1] is False
    assert merged[0][0].y1==844
    assert merged[1][0].y1==879


def test_strong_processor_splits_one_fast_track_when_text_content_changes():
    WORK.mkdir(parents=True,exist_ok=True)
    src=WORK/"content_change_one_track.mp4"
    w=cv2.VideoWriter(str(src),cv2.VideoWriter_fourcc(*"mp4v"),10.0,(360,200))
    assert w.isOpened()
    dets=[]
    for i in range(12):
        f=np.full((200,360,3),105,np.uint8)
        xs=(85,125,165,205) if i<6 else (105,150,195,240,285)
        for x in xs:
            _outlined_rect(f,x,80,x+24,120)
        w.write(f)
        dets.append(FrameDetections(
            i,i/10.0,
            [Candidate(np.array([70,70,300,130],np.float32),.96,"HIGH","fake")],
            [],
        ))
    w.release()

    tracks,metrics=build_v5_strong_tracks(
        src,dets,frame_width=360,frame_height=200,
        max_internal_gap=2,bootstrap_count=5,sample_count=5,
    )

    assert metrics["content_split_track_count"]==2
    assert metrics["segment_count"]==2
    assert len(tracks)==2
    assert tracks[0].sorted_frames()==list(range(0,6))
    assert tracks[1].sorted_frames()==list(range(6,12))
    a=tracks[0].observations[0].bbox
    b=tracks[1].observations[6].bbox
    assert not np.allclose(a,b)


def test_coarse_slot_keeps_leftmost_observed_fast_support_for_occlusion_guard():
    from src.video_text.v5_processor import coarse_supported_slots_from_tracks
    from src.video_text.v5_segments import build_subtitle_segments
    from src.video_text.types import SubtitleTrack, TrackObservation

    t=SubtitleTrack(10,confirmed=True)
    for f in range(7):
        x1=130 if f==6 else 160
        t.observations[f]=TrackObservation(
            f,np.array([x1,849,580,903],np.float32),.95,"HIGH"
        )
    segment=build_subtitle_segments([t])[0]

    slots=coarse_supported_slots_from_tracks(
        segment,{10:t},frame_width=720,frame_height=1280
    )

    assert len(slots)==1
    # Center/layout stay robust, but the guard remembers that FAST genuinely
    # supported the farther-left glyph in at least one frame.
    assert slots[0].expected_x1 == 130
    assert 360 <= slots[0].expected_x_center <= 380
