import numpy as np

from src.video_text.types import SubtitleTrack, TrackObservation
from src.video_text.v5_segments import build_subtitle_segments


def track(track_id, rows):
    t=SubtitleTrack(track_id,confirmed=True)
    for f,b in rows:
        t.observations[f]=TrackObservation(
            f,np.array(b,np.float32),.95,"HIGH"
        )
    return t


def test_overlapping_top_and_bottom_line_tracks_form_one_segment():
    top=track(1,[(f,[80,820,640,875]) for f in range(10,21)])
    bottom=track(2,[(f,[300,880,410,930]) for f in range(10,21)])

    segments=build_subtitle_segments([top,bottom])

    assert len(segments)==1
    s=segments[0]
    assert s.start_frame==10
    assert s.end_frame==20
    assert s.track_ids==[1,2]
    assert s.search_bbox[1] <= 820
    assert s.search_bbox[3] >= 930


def test_adjacent_track_fragments_with_same_geometry_are_merged():
    a=track(1,[(f,[100,845,600,905]) for f in range(10,16)])
    b=track(2,[(f,[102,846,598,904]) for f in range(16,22)])

    segments=build_subtitle_segments([a,b])

    assert len(segments)==1
    assert segments[0].start_frame==10
    assert segments[0].end_frame==21


def test_adjacent_different_subtitles_are_not_merged():
    a=track(1,[(f,[100,845,600,905]) for f in range(10,16)])
    b=track(2,[(f,[280,845,430,905]) for f in range(16,22)])

    segments=build_subtitle_segments([a,b])

    assert len(segments)==2


def test_segment_search_box_expands_downward_to_find_missed_second_line():
    a=track(1,[(f,[85,822,635,877]) for f in range(100,111)])

    s=build_subtitle_segments([a])[0]

    # A top-line-only detector box must still give bootstrap enough vertical
    # room to discover a second line below.
    assert s.search_bbox[3] >= 877 + 90


def test_adjacent_tracks_with_large_persistent_edge_shift_are_new_subtitles():
    # Same vertical lane and high IoU, but both horizontal edges jump by about
    # one character height. This is the real 102-107s failure mode: V5 must
    # start a new subtitle segment instead of making one oversized canonical box.
    a=track(1,[(f,[165,849,557,903]) for f in range(10,21)])
    b=track(2,[(f,[108,845,611,903]) for f in range(21,31)])

    segments=build_subtitle_segments([a,b])

    assert len(segments)==2
    assert segments[0].end_frame==20
    assert segments[1].start_frame==21
