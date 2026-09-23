from pathlib import Path

import cv2
import numpy as np

from src.video_text.types import SubtitleTrack, TrackObservation
from src.video_text.v1_processor import (
    apply_v1_static_geometry_lock,
    guarded_ocr_tighten_bbox,
    merge_v1_same_content_tracks,
    tighten_v1_static_tracks_with_ocr,
)


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "tests/video_text/.tmp_v1_processor"


def _outlined_glyph_band(frame, x1, y1, x2, y2):
    cv2.rectangle(frame, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (8, 8, 8), -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (245, 245, 245), -1)
    for x in range(x1 + 18, x2, 30):
        cv2.rectangle(frame, (x, y1 - 2), (x + 8, y2 + 2), (110, 110, 110), -1)


def _make_two_line_video(path, n=12):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (360, 240)
    )
    assert writer.isOpened()
    for _ in range(n):
        frame = np.full((240, 360, 3), 110, np.uint8)
        _outlined_glyph_band(frame, 78, 116, 282, 158)
        _outlined_glyph_band(frame, 160, 166, 200, 206)
        writer.write(frame)
    writer.release()


def _track(track_id, boxes, level="V55_LINE"):
    t = SubtitleTrack(track_id, confirmed=True)
    for fi, box in enumerate(boxes):
        t.observations[fi] = TrackObservation(
            fi, np.asarray(box, np.float32), .95, level, reconstructed=False
        )
    return t


def test_static_two_line_tracks_become_fully_locked_and_non_overlapping():
    src = WORK / "static_two_line.mp4"
    _make_two_line_video(src)

    top_boxes = [
        [72 + (i % 5), 110 + (i % 3), 288 - (i % 2), 164]
        for i in range(12)
    ]
    bottom_boxes = [
        [154 + (i % 3), 160, 206 - (i % 2), 212 - (i % 3)]
        for i in range(12)
    ]
    top = _track(1, top_boxes)
    bottom = _track(2, bottom_boxes)

    metrics = apply_v1_static_geometry_lock(
        src,
        [top, bottom],
        {1: (7, 0), 2: (7, 1)},
        sample_count=9,
        pad_px=2,
    )

    top_after = [
        tuple(np.round(top.observations[f].bbox, 3))
        for f in top.sorted_frames()
    ]
    bottom_after = [
        tuple(np.round(bottom.observations[f].bbox, 3))
        for f in bottom.sorted_frames()
    ]
    assert len(set(top_after)) == 1
    assert len(set(bottom_after)) == 1
    assert top_after[0][3] <= bottom_after[0][1]
    assert metrics["static_locked_track_count"] == 2
    assert metrics["motion_compensated_track_count"] == 0
    assert metrics["postlock_max_edge_range_px"] == 0.0
    assert metrics["prelock_max_edge_range_px"] >= 4.0


def test_static_lock_applies_pairwise_seams_when_three_line_fragments_have_no_global_common_frame():
    from src.video_text.v55_geometry import line_overlap_area

    src = WORK / "pairwise_seams.mp4"
    src.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (360, 240)
    )
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.full((240, 360, 3), 110, np.uint8))
    writer.release()

    middle = _track_at(701, 0, 9, [80, 120, 280, 180])
    early_top = _track_at(702, 3, 4, [150, 105, 210, 130])
    late_bottom = _track_at(703, 8, 9, [150, 170, 210, 205])
    identity = {
        702: (77, 0),
        701: (77, 1),
        703: (77, 2),
    }

    metrics = apply_v1_static_geometry_lock(
        src,
        [middle, early_top, late_bottom],
        identity,
        sample_count=5,
        pad_px=2,
    )

    for fi in (3, 4):
        assert line_overlap_area(
            early_top.observations[fi].bbox,
            middle.observations[fi].bbox,
        ) == 0.0
    for fi in (8, 9):
        assert line_overlap_area(
            middle.observations[fi].bbox,
            late_bottom.observations[fi].bbox,
        ) == 0.0
    assert metrics["final_overlap_frame_count"] == 0
    assert metrics["temporal_seam_adjustment_count"] >= 2


def test_static_lock_keeps_short_wrapped_second_line_short():
    src = WORK / "short_second_line.mp4"
    _make_two_line_video(src)
    top = _track(10, [[70, 110, 290, 164] for _ in range(12)])
    bottom = _track(11, [[154, 160, 206, 212] for _ in range(12)])

    apply_v1_static_geometry_lock(
        src,
        [top, bottom],
        {10: (8, 0), 11: (8, 1)},
        sample_count=9,
        pad_px=2,
    )

    tb = top.observations[0].bbox
    bb = bottom.observations[0].bbox
    assert (bb[2] - bb[0]) < .35 * (tb[2] - tb[0])
    assert abs((tb[0] + tb[2]) * .5 - (bb[0] + bb[2]) * .5) <= 3.0


def test_true_translation_track_keeps_motion_but_size_is_locked():
    src = WORK / "moving.mp4"
    path = src
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (360, 240)
    )
    assert writer.isOpened()
    boxes = []
    for i in range(10):
        frame = np.full((240, 360, 3), 110, np.uint8)
        _outlined_glyph_band(frame, 70 + i, 120, 270 + i, 166)
        writer.write(frame)
        boxes.append([68 + i, 118, 272 + i, 168])
    writer.release()

    track = _track(20, boxes)
    metrics = apply_v1_static_geometry_lock(
        src, [track], {20: (9, 0)}, sample_count=7, pad_px=2
    )
    after = [track.observations[f].bbox for f in track.sorted_frames()]
    widths = [round(float(b[2] - b[0]), 4) for b in after]
    assert metrics["motion_compensated_track_count"] == 1
    assert len(set(widths)) == 1
    assert after[-1][0] > after[0][0] + 5


def _track_at(track_id, start, end, box, level="V55_LINE"):
    t = SubtitleTrack(track_id, confirmed=True)
    for fi in range(start, end + 1):
        t.observations[fi] = TrackObservation(
            fi, np.asarray(box, np.float32), .95, level, reconstructed=False
        )
    return t


def test_same_content_adjacent_segments_merge_into_one_lifecycle():
    a = _track_at(101, 0, 5, [70, 110, 290, 180])
    b = _track_at(102, 6, 10, [80, 120, 280, 170])
    tracks, identity, compact, metrics = merge_v1_same_content_tracks(
        None,
        [a, b],
        {101: (1, 0), 102: (2, 0)},
        recognizer=None,
        text_evidence={
            101: ["你说得好听"],
            102: ["你说得好听"],
        },
        visual_evidence={(101, 102): .82},
    )
    assert [t.track_id for t in tracks] == [101]
    assert tracks[0].sorted_frames() == list(range(0, 11))
    assert identity == {101: (1, 0)}
    assert len(compact[101]) == 2
    assert metrics["merged_group_count"] == 1
    assert metrics["merged_track_count"] == 1


def test_high_visual_similarity_does_not_merge_different_text():
    a = _track_at(111, 0, 8, [80, 120, 280, 170])
    b = _track_at(112, 9, 17, [80, 120, 280, 170])
    tracks, identity, compact, metrics = merge_v1_same_content_tracks(
        None,
        [a, b],
        {111: (3, 0), 112: (4, 0)},
        recognizer=None,
        text_evidence={
            111: ["今天开始全部停药"],
            112: ["我联系了县农技站"],
        },
        visual_evidence={(111, 112): .99},
    )
    assert [t.track_id for t in tracks] == [111, 112]
    assert set(identity) == {111, 112}
    assert compact == {}
    assert metrics["merged_track_count"] == 0


def test_short_same_content_fragment_can_merge_with_mild_ocr_error():
    a = _track_at(121, 0, 40, [70, 115, 290, 175])
    b = _track_at(122, 41, 44, [80, 120, 280, 170])
    tracks, identity, compact, metrics = merge_v1_same_content_tracks(
        None,
        [a, b],
        {121: (5, 0), 122: (6, 0)},
        recognizer=None,
        text_evidence={
            121: ["你谠的妤听"],
            122: ["你说得好听"],
        },
        visual_evidence={(121, 122): .75},
    )
    assert [t.track_id for t in tracks] == [121]
    assert metrics["merged_group_count"] == 1


def test_multiline_subtitle_tracks_are_not_cross_merged():
    top = _track_at(131, 0, 5, [70, 90, 290, 135])
    bottom = _track_at(132, 0, 5, [155, 140, 205, 185])
    nxt = _track_at(133, 6, 12, [80, 120, 280, 170])
    tracks, identity, compact, metrics = merge_v1_same_content_tracks(
        None,
        [top, bottom, nxt],
        {131: (7, 0), 132: (7, 1), 133: (8, 0)},
        recognizer=None,
        text_evidence={
            131: ["同样内容"],
            132: ["下"],
            133: ["同样内容"],
        },
        visual_evidence={(131, 133): .95},
    )
    assert {t.track_id for t in tracks} == {131, 132, 133}
    assert metrics["merged_group_count"] == 0


def test_merged_geometry_does_not_choose_smallest_member_when_it_clips_glyphs():
    src = WORK / "merged_compact.mp4"
    src.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (360, 240)
    )
    assert writer.isOpened()
    for _ in range(10):
        frame = np.full((240, 360, 3), 110, np.uint8)
        _outlined_glyph_band(frame, 80, 120, 280, 166)
        writer.write(frame)
    writer.release()

    track = _track_at(141, 0, 9, [70, 110, 290, 180])
    compact = {
        141: [
            np.asarray([70, 110, 290, 180], np.float32),
            # Deliberately smaller but wrong: it cuts both edge glyphs.
            np.asarray([100, 120, 260, 170], np.float32),
        ]
    }
    metrics = apply_v1_static_geometry_lock(
        src,
        [track],
        {141: (9, 0)},
        sample_count=7,
        pad_px=2,
        compact_candidates=compact,
    )
    box = track.observations[0].bbox
    assert box[0] <= 80
    assert box[2] >= 280
    assert metrics["merged_compact_track_count"] == 0


def test_ocr_tighten_accepts_same_text_and_shrinks_with_safety_pad():
    anchor=np.asarray([100,120,300,180],np.float32)
    candidate=np.asarray([112,128,288,172],np.float32)
    out=guarded_ocr_tighten_bbox(
        anchor,
        "你说得好听",
        candidate,
        "你说得好听",
        safety_pad=2,
    )
    assert np.allclose(out,[110,126,290,174])


def test_ocr_tighten_rejects_candidate_that_loses_leading_glyph():
    anchor=np.asarray([96,851,624,901],np.float32)
    candidate=np.asarray([138,841,634,913],np.float32)
    out=guarded_ocr_tighten_bbox(
        anchor,
        "一亩地往年还能收2分钱",
        candidate,
        "亩地往年还能收2分钱",
        safety_pad=2,
    )
    assert np.allclose(out,anchor)


def test_ocr_tighten_rejects_different_text_even_if_geometry_is_tighter():
    anchor=np.asarray([100,120,300,180],np.float32)
    candidate=np.asarray([120,130,280,170],np.float32)
    out=guarded_ocr_tighten_bbox(
        anchor,
        "今天开始全部停药",
        candidate,
        "我联系了县农技站",
        safety_pad=2,
    )
    assert np.allclose(out,anchor)


class _FakeReader:
    def readtext(self, crop, detail=1, paragraph=False):
        h, w = crop.shape[:2]
        return [
            (
                [[35, 25], [w - 35, 25], [w - 35, h - 25], [35, h - 25]],
                "你说得好听",
                .95,
            )
        ]


class _FakeRecognizer:
    def __init__(self):
        self.reader = _FakeReader()

    def recognize(self, crop):
        return "你说得好听", .95


def test_ocr_track_tightener_keeps_one_static_bbox_and_reduces_area():
    src = WORK / "ocr_tighten.mp4"
    src.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (360, 240)
    )
    assert writer.isOpened()
    for _ in range(8):
        frame = np.full((240, 360, 3), 110, np.uint8)
        _outlined_glyph_band(frame, 90, 125, 270, 165)
        writer.write(frame)
    writer.release()

    track = _track_at(151, 0, 7, [70, 110, 290, 180])
    before = float((290 - 70) * (180 - 110))
    metrics = tighten_v1_static_tracks_with_ocr(
        src,
        [track],
        {151: (10, 0)},
        _FakeRecognizer(),
        min_height_px=58,
        sample_count=3,
    )
    boxes = {
        tuple(np.round(track.observations[f].bbox, 3))
        for f in track.sorted_frames()
    }
    assert len(boxes) == 1
    box = next(iter(boxes))
    after = float((box[2] - box[0]) * (box[3] - box[1]))
    assert after < before
    assert metrics["ocr_tightened_track_count"] == 1
    assert metrics["ocr_tightening_area_ratio_median"] < 1.0

def test_sandwiched_one_frame_transition_fragment_is_absorbed_into_next_episode():
    from src.video_text.v1_processor import absorb_v1_transition_fragments

    prev = _track_at(201, 0, 9, [145, 120, 215, 180])
    transient = _track_at(202, 10, 10, [100, 96, 260, 202])
    nxt = _track_at(203, 11, 30, [85, 112, 275, 182])
    tracks, identity, metrics = absorb_v1_transition_fragments(
        [prev, transient, nxt],
        {201: (20, 0), 202: (21, 0), 203: (22, 0)},
        max_duration=2,
        min_height_inflation=1.25,
    )
    assert [t.track_id for t in tracks] == [201, 203]
    assert tracks[1].sorted_frames()[0] == 10
    assert tracks[1].sorted_frames()[-1] == 30
    assert identity == {201: (20, 0), 203: (22, 0)}
    assert metrics["transition_fragment_absorbed_count"] == 1


def test_short_real_track_is_not_absorbed_when_not_geometrically_inflated():
    from src.video_text.v1_processor import absorb_v1_transition_fragments

    prev = _track_at(211, 0, 9, [145, 120, 215, 180])
    short = _track_at(212, 10, 10, [120, 120, 240, 180])
    nxt = _track_at(213, 11, 30, [85, 112, 275, 182])
    tracks, identity, metrics = absorb_v1_transition_fragments(
        [prev, short, nxt],
        {211: (23, 0), 212: (24, 0), 213: (25, 0)},
        max_duration=2,
        min_height_inflation=1.25,
    )
    assert [t.track_id for t in tracks] == [211, 212, 213]
    assert metrics["transition_fragment_absorbed_count"] == 0


class _SameTextRecognizer:
    def recognize(self, crop):
        return "你说得好听", .95


def test_temporal_glyph_tightener_shrinks_loose_static_box_without_jitter():
    from src.video_text.v1_processor import (
        tighten_v1_static_tracks_with_temporal_glyphs,
    )

    src = WORK / "glyph_tighten.mp4"
    src.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (360, 240)
    )
    assert writer.isOpened()
    for i in range(15):
        frame = np.full((240, 360, 3), 110, np.uint8)
        _outlined_glyph_band(frame, 80, 120, 280, 166)
        if i == 7:
            cv2.rectangle(frame, (35, 105), (55, 195), (245, 245, 245), -1)
        writer.write(frame)
    writer.release()

    track = _track_at(221, 0, 14, [55, 100, 310, 195])
    before = float((310 - 55) * (195 - 100))
    metrics = tighten_v1_static_tracks_with_temporal_glyphs(
        src,
        [track],
        {221: (26, 0)},
        recognizer=_SameTextRecognizer(),
        sample_count=15,
        safety_pad=4,
    )
    boxes = {
        tuple(np.round(track.observations[f].bbox, 3))
        for f in track.sorted_frames()
    }
    assert len(boxes) == 1
    box = next(iter(boxes))
    after = float((box[2] - box[0]) * (box[3] - box[1]))
    assert after < .85 * before
    assert box[0] <= 78 and box[2] >= 282
    assert box[1] <= 118 and box[3] >= 168
    assert metrics["glyph_tightened_track_count"] == 1
    assert metrics["glyph_tightening_area_ratio_median"] < .85


def test_temporal_glyph_tightener_shrinks_without_recognizer_and_ignores_transient_noise():
    from src.video_text.v1_processor import (
        tighten_v1_static_tracks_with_temporal_glyphs,
    )

    src = WORK / "glyph_tighten_no_ocr.mp4"
    src.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (360, 240)
    )
    assert writer.isOpened()
    for i in range(15):
        frame = np.full((240, 360, 3), 110, np.uint8)
        _outlined_glyph_band(frame, 80, 120, 280, 166)
        if i == 7:
            cv2.rectangle(frame, (30, 102), (52, 198), (245, 245, 245), -1)
        writer.write(frame)
    writer.release()

    track = _track_at(225, 0, 14, [55, 105, 310, 190])
    old_area = float((310 - 55) * (190 - 105))
    metrics = tighten_v1_static_tracks_with_temporal_glyphs(
        src, [track], {225: (260, 0)}, recognizer=None,
        sample_count=15, safety_pad=4,
    )
    boxes = {
        tuple(np.round(track.observations[f].bbox, 3))
        for f in track.sorted_frames()
    }
    assert len(boxes) == 1
    box = next(iter(boxes))
    new_area = float((box[2] - box[0]) * (box[3] - box[1]))
    assert new_area < .98 * old_area
    assert box[0] <= 78 and box[2] >= 282
    assert box[1] <= 118 and box[3] >= 168
    assert box[0] > 52
    assert metrics["glyph_tightened_track_count"] == 1


def test_temporal_glyph_tightener_without_ocr_keeps_persistent_thin_leading_glyph():
    from src.video_text.v1_processor import (
        tighten_v1_static_tracks_with_temporal_glyphs,
    )

    src = WORK / "glyph_tighten_thin_no_ocr.mp4"
    src.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (360, 240)
    )
    assert writer.isOpened()
    for _ in range(11):
        frame = np.full((240, 360, 3), 110, np.uint8)
        cv2.rectangle(frame, (70, 139), (108, 144), (245, 245, 245), -1)
        cv2.rectangle(frame, (68, 137), (110, 146), (8, 8, 8), 2)
        _outlined_glyph_band(frame, 122, 120, 280, 166)
        writer.write(frame)
    writer.release()

    track = _track_at(226, 0, 10, [55, 100, 310, 195])
    tighten_v1_static_tracks_with_temporal_glyphs(
        src, [track], {226: (261, 0)}, recognizer=None,
        sample_count=11, safety_pad=4,
    )
    box = track.observations[0].bbox
    assert box[0] <= 68
    assert box[2] >= 282


def test_visual_only_adjacent_segments_merge_only_at_high_similarity():
    a = _track_at(227, 0, 5, [80, 120, 280, 170])
    b = _track_at(228, 6, 11, [82, 121, 282, 171])
    tracks, _, _, metrics = merge_v1_same_content_tracks(
        None, [a, b], {227: (262, 0), 228: (263, 0)},
        recognizer=None,
        text_evidence={227: [], 228: []},
        visual_evidence={(227, 228): .96},
    )
    assert [t.track_id for t in tracks] == [227]
    assert metrics["merged_track_count"] == 1

    c = _track_at(229, 0, 5, [80, 120, 280, 170])
    d = _track_at(230, 6, 11, [82, 121, 282, 171])
    tracks, _, _, metrics = merge_v1_same_content_tracks(
        None, [c, d], {229: (264, 0), 230: (265, 0)},
        recognizer=None,
        text_evidence={229: [], 230: []},
        visual_evidence={(229, 230): .90},
    )
    assert [t.track_id for t in tracks] == [229, 230]
    assert metrics["merged_track_count"] == 0


def test_temporal_glyph_tightener_falls_back_when_candidate_loses_text():
    from src.video_text.v1_processor import (
        tighten_v1_static_tracks_with_temporal_glyphs,
    )

    class RejectingRecognizer:
        def recognize(self, crop):
            h, w = crop.shape[:2]
            if w >= 255 and h >= 95:
                return "完整字幕", .9
            return "字", .9

    src = WORK / "glyph_guard.mp4"
    src.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(src), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (360, 240)
    )
    assert writer.isOpened()
    for _ in range(9):
        frame = np.full((240, 360, 3), 110, np.uint8)
        _outlined_glyph_band(frame, 80, 120, 280, 166)
        writer.write(frame)
    writer.release()

    track = _track_at(231, 0, 8, [55, 100, 310, 195])
    anchor = track.observations[0].bbox.copy()
    metrics = tighten_v1_static_tracks_with_temporal_glyphs(
        src,
        [track],
        {231: (27, 0)},
        recognizer=RejectingRecognizer(),
        sample_count=9,
        safety_pad=4,
    )
    assert np.allclose(track.observations[0].bbox, anchor)
    assert metrics["glyph_tightened_track_count"] == 0



def test_ocr_tighten_rejects_same_length_leading_glyph_substitution():
    anchor=np.asarray([88,848,631,904],np.float32)
    candidate=np.asarray([102,848,631,904],np.float32)
    out=guarded_ocr_tighten_bbox(
        anchor,
        "还在我家借稻米下锅吧",
        candidate,
        "丕在我家借稻米下锅吧",
        safety_pad=0,
    )
    assert np.allclose(out,anchor)


def test_glyph_guard_rejects_same_length_leading_character_mutation():
    from src.video_text.v1_processor import _glyph_candidate_preserves_text

    class WidthSensitiveRecognizer:
        def recognize(self,crop):
            if crop.shape[1] >= 100:
                return "还在我家借稻米下锅吧",.95
            return "丕在我家借稻米下锅吧",.95

    frame=np.zeros((80,140,3),np.uint8)
    frame_map={1:frame,2:frame,3:frame}
    anchor=np.asarray([10,10,120,60],np.float32)
    candidate=np.asarray([24,10,120,60],np.float32)
    assert not _glyph_candidate_preserves_text(
        frame_map,[1,2,3],anchor,candidate,WidthSensitiveRecognizer()
    )


def test_temporal_row_component_selection_keeps_fragment_that_overlaps_text_band():
    from src.video_text.v1_processor import _select_temporal_text_components

    height=56
    comps=[
        [58,5,106,54,513,74.9,23.0,49,48],
        [113,5,161,54,986,135.5,29.6,49,48],
        [4,22,52,54,338,21.4,42.4,32,48],
        [7,5,16,15,47,11.4,9.3,10,9],
    ]
    keep=_select_temporal_text_components(comps,height)
    assert any(int(c[0])==4 for c in keep)


def test_edgewise_glyph_guard_finds_tightest_safe_top_between_anchor_and_candidate():
    from src.video_text.v1_processor import _guard_temporal_candidate_edges

    class EdgeSensitiveRecognizer:
        def recognize(self,crop):
            h,w=crop.shape[:2]
            if h < 55:
                return "金村跟着你把虾种放进水",.8
            return "全村跟着你把虾种放进水",.9

    frame=np.zeros((90,140,3),np.uint8)
    frame_map={1:frame,2:frame,3:frame}
    anchor=np.asarray([10,10,120,70],np.float32)
    candidate=np.asarray([20,20,110,70],np.float32)
    out=_guard_temporal_candidate_edges(
        frame_map,[1,2,3],anchor,candidate,EdgeSensitiveRecognizer()
    )
    assert np.allclose(out,[20,15,110,70])



def test_single_line_height_regularizer_shrinks_only_high_outlier_and_preserves_center():
    from src.video_text.v1_processor import regularize_v1_single_line_height

    tracks=[]
    identity={}
    heights=[52,54,54,56,54,52,55,70]
    for i,h in enumerate(heights,300):
        y1=876-h/2
        y2=876+h/2
        t=_track_at(i,0,9,[100,y1,300,y2])
        tracks.append(t)
        identity[i]=(i,0)

    metrics=regularize_v1_single_line_height(
        tracks,
        identity,
        frame_height=1280,
        min_reference_tracks=6,
        safety_extra_px=4,
    )
    out=tracks[-1].observations[0].bbox
    assert round(float(out[3]-out[1]),3)==58.0
    assert round(float((out[1]+out[3])*.5),3)==876.0
    assert metrics["height_regularized_track_count"]==1
    for t,h in zip(tracks[:-1],heights[:-1]):
        b=t.observations[0].bbox
        assert round(float(b[3]-b[1]),3)==float(h)


def test_single_line_height_regularizer_does_not_touch_normal_56px_frame45_style():
    from src.video_text.v1_processor import regularize_v1_single_line_height

    tracks=[]
    identity={}
    for i,h in enumerate([52,54,54,56,54,52,55,56],400):
        t=_track_at(i,0,9,[88,876-h/2,631,876+h/2])
        tracks.append(t)
        identity[i]=(i,0)
    metrics=regularize_v1_single_line_height(
        tracks,identity,frame_height=1280,min_reference_tracks=6
    )
    assert metrics["height_regularized_track_count"]==0
    assert np.allclose(
        tracks[-1].observations[0].bbox,
        np.asarray([88,848,631,904],np.float32),
    )


def test_height_regularizer_ignores_multiline_roles():
    from src.video_text.v1_processor import regularize_v1_single_line_height

    refs=[]
    identity={}
    for i in range(8):
        tid=500+i
        t=_track_at(tid,0,9,[100,849,300,903])
        refs.append(t)
        identity[tid]=(tid,0)

    top=_track_at(600,0,9,[80,810,640,880])
    bottom=_track_at(601,0,9,[320,880,400,940])
    identity[600]=(999,0)
    identity[601]=(999,1)
    before_top=top.observations[0].bbox.copy()
    before_bottom=bottom.observations[0].bbox.copy()

    regularize_v1_single_line_height(
        refs+[top,bottom],identity,frame_height=1280,min_reference_tracks=6
    )
    assert np.allclose(top.observations[0].bbox,before_top)
    assert np.allclose(bottom.observations[0].bbox,before_bottom)


def test_edgewise_guard_defers_to_temporal_geometry_when_ocr_is_low_confidence():
    from src.video_text.v1_processor import _guard_temporal_candidate_edges

    class LowConfidenceRecognizer:
        def recognize(self,crop):
            # Narrower crops mutate the first CJK character, but both reads
            # are far below a confidence level suitable for geometry veto.
            if crop.shape[1] < 118:
                return "痧见谁把害虫养进田里",.08
            return "没见谁把害虫养进田里",.10

    frame=np.zeros((90,160,3),np.uint8)
    frame_map={1:frame,2:frame,3:frame}
    anchor=np.asarray([10,10,130,70],np.float32)
    candidate=np.asarray([16,10,130,68],np.float32)
    out=_guard_temporal_candidate_edges(
        frame_map,[1,2,3],anchor,candidate,LowConfidenceRecognizer()
    )
    assert np.allclose(out,candidate)
