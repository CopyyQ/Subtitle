import numpy as np
from src.video_text.types import SubtitleTrack, TrackObservation
from src.video_text.lifecycle import reconstruct_track

def track_with(items, levels=None):
    t=SubtitleTrack(1,confirmed=True)
    levels=levels or {}
    for f,b in items.items():
        level=levels.get(f,"HIGH")
        t.observations[f]=TrackObservation(f,np.array(b,np.float32),.9,level)
    return t

def event_types(events):
    return [e.event_type for e in events]

def test_one_frame_gap_between_same_track_is_reconstructed():
    t=track_with({10:[100,800,400,850],12:[102,800,402,850]})
    out,events=reconstruct_track(t,max_gap=2)
    assert 11 in out.observations
    assert out.observations[11].reconstructed is True
    assert "internal_miss_recovered" in event_types(events)

def test_two_frame_gap_is_linearly_interpolated():
    t=track_with({10:[100,800,400,850],13:[103,803,403,853]})
    out,events=reconstruct_track(t,max_gap=2)
    assert np.allclose(out.observations[11].bbox,[101,801,401,851])
    assert np.allclose(out.observations[12].bbox,[102,802,402,852])
    assert sum(e.event_type=="internal_miss_recovered" for e in events)==1

def test_real_track_end_creates_no_ghost_boxes():
    t=track_with({68:[100,800,400,850],69:[100,800,400,850],70:[100,800,400,850]})
    out,events=reconstruct_track(t,max_gap=2,total_frames=74)
    assert 71 not in out.observations
    assert 72 not in out.observations
    assert 73 not in out.observations
    assert "track_end" in event_types(events)

def test_low_backfill_event_is_reported():
    t=track_with({20:[100,800,400,850],21:[101,800,401,850],22:[102,800,402,850]},
                 levels={20:"LOW",21:"HIGH",22:"HIGH"})
    out,events=reconstruct_track(t,max_gap=2)
    assert "track_start_backfill" in event_types(events)


def test_reconstructed_gap_is_dropped_when_actual_same_line_box_already_covers_it():
    from src.video_text.lifecycle import reconstruct_tracks, suppress_reconstructed_overlaps

    small=SubtitleTrack(48,confirmed=True)
    small.observations[132]=TrackObservation(
        132,np.array([329,875,397,930],np.float32),.95,"HIGH"
    )
    small.observations[134]=TrackObservation(
        134,np.array([329,875,397,930],np.float32),.95,"HIGH"
    )

    actual=SubtitleTrack(5,confirmed=True)
    actual.observations[133]=TrackObservation(
        133,np.array([191,875,386,930],np.float32),.95,"HIGH"
    )

    tracks,events=reconstruct_tracks([small,actual],max_gap=2,total_frames=140)
    assert tracks[0].observations[133].reconstructed is True
    assert any(
        e.event_type=="internal_miss_recovered"
        and e.track_id==48 and e.start_frame==133
        for e in events
    )

    tracks,events,suppressed=suppress_reconstructed_overlaps(tracks,events)
    small_out=next(t for t in tracks if t.track_id==48)
    assert 133 not in small_out.observations
    assert suppressed==1
    assert not any(
        e.event_type=="internal_miss_recovered"
        and e.track_id==48 and e.start_frame<=133<=e.end_frame
        for e in events
    )


def test_reconstructed_gap_is_kept_when_no_actual_box_covers_it():
    from src.video_text.lifecycle import reconstruct_tracks, suppress_reconstructed_overlaps

    t=SubtitleTrack(1,confirmed=True)
    t.observations[10]=TrackObservation(
        10,np.array([100,800,400,850],np.float32),.95,"HIGH"
    )
    t.observations[12]=TrackObservation(
        12,np.array([102,800,402,850],np.float32),.95,"HIGH"
    )
    tracks,events=reconstruct_tracks([t],max_gap=2,total_frames=20)
    tracks,events,suppressed=suppress_reconstructed_overlaps(tracks,events)
    assert tracks[0].observations[11].reconstructed is True
    assert suppressed==0
    assert any(
        e.event_type=="internal_miss_recovered"
        and e.start_frame==11 and e.end_frame==11
        for e in events
    )
