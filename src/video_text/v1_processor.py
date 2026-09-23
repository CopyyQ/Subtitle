from __future__ import annotations

import cv2
import numpy as np

from .chinese_validator import contains_cjk
from .v5_processor import _read_frames
from .v5_content_split import _mask_similarity
from .v5_slots import _text_core_mask
from .v55_geometry import estimate_separator_seam, line_overlap_area
from .v1_geometry import detect_consistent_translation, temporal_canonical_bbox


def _sample_ids(frame_ids, count):
    frame_ids = list(frame_ids)
    if not frame_ids:
        return []
    if len(frame_ids) <= int(count):
        return list(frame_ids)
    positions = np.linspace(0, len(frame_ids) - 1, int(count))
    return sorted({frame_ids[int(round(p))] for p in positions})


def _median_filter(values, window=3):
    values = np.asarray(values, dtype=np.float32)
    radius = max(0, int(window) // 2)
    out = []
    for i in range(len(values)):
        lo = max(0, i - radius)
        hi = min(len(values), i + radius + 1)
        out.append(float(np.median(values[lo:hi])))
    return np.asarray(out, dtype=np.float32)


def _track_edge_range(track):
    fs = track.sorted_frames()
    if len(fs) < 2:
        return 0.0
    arr = np.stack(
        [track.observations[fi].bbox for fi in fs],
        axis=0,
    ).astype(np.float32)
    return float(np.max(np.ptp(arr, axis=0)))


def _lock_moving_track(track):
    fs = track.sorted_frames()
    arr = np.stack(
        [track.observations[fi].bbox for fi in fs],
        axis=0,
    ).astype(np.float32)
    widths = arr[:, 2] - arr[:, 0]
    heights = arr[:, 3] - arr[:, 1]
    width = float(np.median(widths))
    height = float(np.median(heights))
    cxs = _median_filter((arr[:, 0] + arr[:, 2]) * .5, window=3)
    cys = _median_filter((arr[:, 1] + arr[:, 3]) * .5, window=3)
    for fi, cx, cy in zip(fs, cxs, cys):
        track.observations[fi].bbox = np.array(
            [
                cx - width * .5,
                cy - height * .5,
                cx + width * .5,
                cy + height * .5,
            ],
            dtype=np.float32,
        )


def _lock_static_track(video_path, track, sample_count, pad_px):
    fs = track.sorted_frames()
    sample_ids = _sample_ids(fs, sample_count)
    frame_map = _read_frames(video_path, sample_ids)
    frames = [frame_map[fi] for fi in sample_ids]
    boxes = [track.observations[fi].bbox for fi in sample_ids]
    canonical = temporal_canonical_bbox(
        frames,
        boxes,
        pad_px=pad_px,
    )
    for fi in fs:
        track.observations[fi].bbox = canonical.copy()
    return canonical


def _align_static_sibling_centers(
    ordered_tracks,
    common_frames,
    *,
    frame_width,
    max_shift_px=4.0,
):
    if len(ordered_tracks) < 2 or not common_frames:
        return 0
    centers = []
    for track in ordered_tracks:
        vals = [
            .5 * (
                float(track.observations[fi].bbox[0])
                + float(track.observations[fi].bbox[2])
            )
            for fi in common_frames
            if fi in track.observations
        ]
        if not vals:
            centers.append(None)
        else:
            centers.append(float(np.median(np.asarray(vals, np.float32))))
    valid = [c for c in centers if c is not None]
    if len(valid) < 2:
        return 0

    shared = float(np.median(np.asarray(valid, np.float32)))
    adjusted = 0
    for track, center in zip(ordered_tracks, centers):
        if center is None:
            continue
        shift = float(np.clip(
            shared - center,
            -float(max_shift_px),
            float(max_shift_px),
        ))
        if abs(shift) < 1e-6:
            continue
        for fi in track.sorted_frames():
            box = track.observations[fi].bbox.copy()
            width = float(box[2] - box[0])
            nx1 = float(box[0]) + shift
            nx1 = max(0.0, min(float(frame_width) - width, nx1))
            box[0] = nx1
            box[2] = nx1 + width
            track.observations[fi].bbox = box
        adjusted += 1
    return adjusted


def _fixed_multiline_seam(
    video_path,
    ordered_tracks,
    common_frames,
    *,
    sample_count,
):
    if len(ordered_tracks) < 2 or not common_frames:
        return 0
    sample_ids = _sample_ids(common_frames, sample_count)
    frame_map = _read_frames(video_path, sample_ids)
    adjustments = 0

    for top_track, bottom_track in zip(ordered_tracks, ordered_tracks[1:]):
        overlap_ids = [
            fi for fi in sample_ids
            if fi in top_track.observations and fi in bottom_track.observations
        ]
        if not overlap_ids:
            continue

        top_box = top_track.observations[overlap_ids[0]].bbox.copy()
        bottom_box = bottom_track.observations[overlap_ids[0]].bbox.copy()
        if line_overlap_area(top_box, bottom_box) <= 0:
            continue

        seams = [
            estimate_separator_seam(
                frame_map[fi],
                top_track.observations[fi].bbox,
                bottom_track.observations[fi].bbox,
            )
            for fi in overlap_ids
        ]
        seam = float(np.median(np.asarray(seams, dtype=np.float32)))
        min_seam = float(top_box[1]) + 1.0
        max_seam = float(bottom_box[3]) - 1.0
        seam = max(min_seam, min(max_seam, seam))

        for fi in top_track.sorted_frames():
            box = top_track.observations[fi].bbox.copy()
            box[3] = min(float(box[3]), seam)
            top_track.observations[fi].bbox = box
        for fi in bottom_track.sorted_frames():
            box = bottom_track.observations[fi].bbox.copy()
            box[1] = max(float(box[1]), seam)
            bottom_track.observations[fi].bbox = box
        adjustments += 1

    return adjustments



def _normalized_text(text):
    return "".join(str(text or "").split())


def _edit_similarity(a, b):
    a = _normalized_text(a)
    b = _normalized_text(b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(
                min(
                    cur[-1] + 1,
                    prev[j] + 1,
                    prev[j - 1] + (ca != cb),
                )
            )
        prev = cur
    return 1.0 - float(prev[-1]) / float(max(len(a), len(b)))


def guarded_ocr_tighten_bbox(
    anchor,
    current_text,
    candidate_bbox,
    candidate_text,
    *,
    safety_pad=2.0,
    min_text_similarity=.65,
):
    anchor = np.asarray(anchor, dtype=np.float32).copy()
    candidate = np.asarray(candidate_bbox, dtype=np.float32)
    current = _normalized_text(current_text)
    proposed = _normalized_text(candidate_text)
    if not current or not proposed or not contains_cjk(proposed):
        return anchor
    if len(proposed) < len(current):
        return anchor
    if _edit_similarity(current, proposed) < float(min_text_similarity):
        return anchor

    pad = max(0.0, float(safety_pad))
    out = np.array(
        [
            max(float(anchor[0]), float(candidate[0]) - pad),
            max(float(anchor[1]), float(candidate[1]) - pad),
            min(float(anchor[2]), float(candidate[2]) + pad),
            min(float(anchor[3]), float(candidate[3]) + pad),
        ],
        dtype=np.float32,
    )
    if out[2] <= out[0] or out[3] <= out[1]:
        return anchor

    # Horizontal tightening is only safe if the OCR boundary characters are
    # preserved. Edit similarity alone is insufficient: a clipped leading
    # Chinese glyph can mutate into another valid CJK character while keeping
    # the same string length (e.g. 还 -> 丕), which previously passed the
    # similarity gate and caused visible left-edge clipping.
    if (
        float(out[0]) > float(anchor[0]) + 0.5
        and proposed[0] != current[0]
    ):
        return anchor
    if (
        float(out[2]) < float(anchor[2]) - 0.5
        and proposed[-1] != current[-1]
    ):
        return anchor

    old_area = float(
        (anchor[2] - anchor[0]) * (anchor[3] - anchor[1])
    )
    new_area = float((out[2] - out[0]) * (out[3] - out[1]))
    if new_area >= .98 * max(1.0, old_area):
        return anchor
    return out


def _track_median_bbox(track):
    fs = track.sorted_frames()
    arr = np.stack(
        [track.observations[fi].bbox for fi in fs],
        axis=0,
    ).astype(np.float32)
    return np.median(arr, axis=0).astype(np.float32)


def _collect_track_text_evidence(video_path, tracks, recognizer):
    if recognizer is None:
        return {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    evidence = {}
    try:
        for track in tracks:
            fs = track.sorted_frames()
            if not fs:
                evidence[track.track_id] = []
                continue
            if len(fs) <= 5:
                sample_ids = _sample_ids(fs, min(3, len(fs)))
            else:
                positions = (.25, .50, .75)
                sample_ids = sorted({
                    fs[int(round((len(fs) - 1) * p))]
                    for p in positions
                })
            vals = []
            for fi in sample_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, frame = cap.read()
                if not ok:
                    continue
                b = track.observations[fi].bbox
                h, w = frame.shape[:2]
                x1 = max(0, int(np.floor(float(b[0]))))
                y1 = max(0, int(np.floor(float(b[1]))))
                x2 = min(w, int(np.ceil(float(b[2]))))
                y2 = min(h, int(np.ceil(float(b[3]))))
                if x2 <= x1 or y2 <= y1:
                    continue
                text, _ = recognizer.recognize(frame[y1:y2, x1:x2])
                text = _normalized_text(text)
                if text:
                    vals.append(text)
            evidence[track.track_id] = vals
    finally:
        cap.release()
    return evidence


def _boundary_visual_similarity(video_path, a, b):
    fa = a.sorted_frames()[-1]
    fb = b.sorted_frames()[0]
    ba = a.observations[fa].bbox
    bb = b.observations[fb].bbox
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fa))
        oka, ima = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fb))
        okb, imb = cap.read()
    finally:
        cap.release()
    if not oka or not okb:
        return 0.0
    h, w = ima.shape[:2]
    x1 = max(0, int(np.floor(min(float(ba[0]), float(bb[0])))) - 10)
    y1 = max(0, int(np.floor(min(float(ba[1]), float(bb[1])))) - 10)
    x2 = min(w, int(np.ceil(max(float(ba[2]), float(bb[2])))) + 10)
    y2 = min(h, int(np.ceil(max(float(ba[3]), float(bb[3])))) + 10)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    ma = _text_core_mask(ima[y1:y2, x1:x2])
    mb = _text_core_mask(imb[y1:y2, x1:x2])
    return float(_mask_similarity(ma, mb))


def _max_text_similarity(texts_a, texts_b):
    vals = [
        _edit_similarity(a, b)
        for a in texts_a
        for b in texts_b
        if _normalized_text(a) and _normalized_text(b)
    ]
    return max(vals, default=0.0)


def absorb_v1_transition_fragments(
    tracks,
    identity,
    *,
    max_duration=2,
    min_height_inflation=1.25,
):
    tracks = list(tracks)
    identity = dict(identity)
    by_id = {t.track_id: t for t in tracks}

    sid_members = {}
    for tid, tag in identity.items():
        sid, line_id = tag
        sid_members.setdefault(int(sid), []).append((int(line_id), int(tid)))

    eligible = []
    for track in tracks:
        tag = identity.get(track.track_id)
        if tag is None:
            continue
        sid, line_id = tag
        if int(line_id) != 0 or len(sid_members.get(int(sid), [])) != 1:
            continue
        fs = track.sorted_frames()
        if fs:
            eligible.append((fs[0], fs[-1], track.track_id))
    eligible.sort()

    removed = set()
    absorbed = []
    for i in range(1, len(eligible) - 1):
        p_start, p_end, p_id = eligible[i - 1]
        c_start, c_end, c_id = eligible[i]
        n_start, n_end, n_id = eligible[i + 1]
        if c_id in removed:
            continue
        duration = int(c_end - c_start + 1)
        if duration > int(max_duration):
            continue
        if int(c_start) != int(p_end) + 1:
            continue
        if int(n_start) != int(c_end) + 1:
            continue

        prev = by_id[p_id]
        cur = by_id[c_id]
        nxt = by_id[n_id]
        prev_box = _track_median_bbox(prev)
        cur_box = _track_median_bbox(cur)
        next_box = _track_median_bbox(nxt)
        prev_h = float(prev_box[3] - prev_box[1])
        cur_h = float(cur_box[3] - cur_box[1])
        next_h = float(next_box[3] - next_box[1])
        reference_h = max(prev_h, next_h, 1.0)
        if cur_h < float(min_height_inflation) * reference_h:
            continue

        # Transitional overlays belong to the episode that starts immediately
        # afterwards. This removes one-frame geometry flashes while retaining
        # the visible frame in the output timeline.
        for fi, obs in cur.observations.items():
            nxt.observations[int(fi)] = obs
        removed.add(c_id)
        identity.pop(c_id, None)
        absorbed.append((c_id, n_id))

    out = [t for t in tracks if t.track_id not in removed]
    out.sort(
        key=lambda t: (
            min(t.sorted_frames()) if t.sorted_frames() else 10**12,
            t.track_id,
        )
    )
    return out, identity, {
        "transition_fragment_absorbed_count": len(absorbed),
        "transition_fragment_absorptions": [
            {"fragment_track_id": int(a), "target_track_id": int(b)}
            for a, b in absorbed
        ],
    }



def _select_temporal_text_components(comps, height):
    comps = list(comps)
    if not comps:
        return []

    ys = np.asarray([float(c[6]) for c in comps], dtype=np.float32)
    weights = np.asarray([float(c[4]) for c in comps], dtype=np.float32)
    best_y = None
    best_score = -1.0
    for y0 in ys:
        selected = np.abs(ys - y0) <= max(5.0, .16 * float(height))
        score = float(np.sum(weights[selected]))
        if score > best_score:
            best_score = score
            best_y = float(np.average(ys[selected], weights=weights[selected]))

    band = max(7.0, .18 * float(height))
    row_lo = best_y - band
    row_hi = best_y + band
    keep = []
    for c in comps:
        comp_y1 = float(c[1])
        comp_y2 = float(c[3])
        comp_h = float(c[7])
        overlap = max(
            0.0,
            min(comp_y2, row_hi) - max(comp_y1, row_lo),
        )
        min_overlap = min(4.0, max(1.0, .15 * comp_h))
        if (
            comp_h >= max(3.0, .10 * float(height))
            and overlap >= min_overlap
        ):
            keep.append(c)
    return keep


def _temporal_glyph_candidate(
    frames,
    anchor,
    *,
    safety_pad=4,
    vote_threshold=.25,
):
    frames = list(frames)
    anchor = np.asarray(anchor, dtype=np.float32)
    if not frames:
        return anchor.copy()
    x1, y1, x2, y2 = [int(round(float(v))) for v in anchor]
    if x2 <= x1 or y2 <= y1:
        return anchor.copy()
    height = y2 - y1
    width = x2 - x1
    votes = np.zeros((height, width), dtype=np.float32)

    for frame in frames:
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        local_min = cv2.erode(gray, np.ones((7, 7), np.uint8))
        contrast = (
            gray.astype(np.int16) - local_min.astype(np.int16)
        )
        p70 = float(np.percentile(gray, 70))
        floor = max(115.0, min(175.0, p70))
        core = (
            (gray >= floor)
            & (hsv[:, :, 1] <= 125)
            & (contrast >= 28)
        )
        mask = core.astype(np.uint8) * 255
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((2, 2), np.uint8),
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((3, 3), np.uint8),
        )
        votes += (mask > 0).astype(np.float32)

    persistent = (
        votes / max(1, len(frames)) >= float(vote_threshold)
    ).astype(np.uint8) * 255
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(
        persistent, 8
    )
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        cy = float(centroids[i][1])
        cx = float(centroids[i][0])
        if int(area) < 3 or int(h) < 3 or int(w) < 2:
            continue
        if cy < .15 * height or cy > .85 * height:
            continue
        comps.append(
            [x, y, x + w, y + h, area, cx, cy, h, w]
        )
    if not comps:
        return anchor.copy()

    keep = _select_temporal_text_components(comps, height)
    if not keep:
        return anchor.copy()

    pad = max(0, int(safety_pad))
    xx1 = max(0, min(c[0] for c in keep) - pad)
    yy1 = max(0, min(c[1] for c in keep) - pad)
    xx2 = min(width, max(c[2] for c in keep) + pad)
    yy2 = min(height, max(c[3] for c in keep) + pad)
    candidate = np.asarray(
        [x1 + xx1, y1 + yy1, x1 + xx2, y1 + yy2],
        dtype=np.float32,
    )
    if candidate[2] <= candidate[0] or candidate[3] <= candidate[1]:
        return anchor.copy()
    return candidate



def _guard_temporal_candidate_edges(
    frame_map,
    sample_ids,
    anchor,
    candidate,
    recognizer,
    *,
    min_similarity=.65,
):
    anchor = np.asarray(anchor, dtype=np.float32).copy()
    candidate = np.asarray(candidate, dtype=np.float32)
    if recognizer is None:
        aw=float(anchor[2]-anchor[0])
        ah=float(anchor[3]-anchor[1])
        cw=float(candidate[2]-candidate[0])
        ch=float(candidate[3]-candidate[1])
        if aw<=0 or ah<=0 or cw<=0 or ch<=0:
            return anchor
        old_area=aw*ah
        new_area=cw*ch
        inside=(
            float(candidate[0])>=float(anchor[0])-0.5
            and float(candidate[1])>=float(anchor[1])-0.5
            and float(candidate[2])<=float(anchor[2])+0.5
            and float(candidate[3])<=float(anchor[3])+0.5
        )
        area_ratio=new_area/max(old_area,1e-6)
        if (
            inside
            and cw>=.70*aw
            and ch>=.60*ah
            and .45<=area_ratio<=.98
        ):
            return candidate.copy()
        return anchor

    ids = _sample_ids(sample_ids, min(3, len(sample_ids)))
    if not ids:
        return anchor

    def crop(frame, box):
        h, w = frame.shape[:2]
        x1 = max(0, int(np.floor(float(box[0]))))
        y1 = max(0, int(np.floor(float(box[1]))))
        x2 = min(w, int(np.ceil(float(box[2]))))
        y2 = min(h, int(np.ceil(float(box[3]))))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2]

    anchor_text = {}
    for fi in ids:
        frame = frame_map.get(fi)
        if frame is None:
            continue
        c = crop(frame, anchor)
        if c is None:
            continue
        text, conf = recognizer.recognize(c)
        text = _normalized_text(text)
        if text:
            anchor_text[fi] = (
                text,
                0.0 if conf is None else float(conf),
            )

    if not anchor_text:
        return anchor

    def trial_is_safe(trial, edge):
        checks = 0
        passed = 0
        for fi, (old_text, old_conf) in anchor_text.items():
            frame = frame_map.get(fi)
            if frame is None:
                continue
            c = crop(frame, trial)
            if c is None:
                continue
            new_text, new_conf = recognizer.recognize(c)
            new_text = _normalized_text(new_text)
            new_conf = 0.0 if new_conf is None else float(new_conf)
            checks += 1
            if (
                not new_text
                or not contains_cjk(new_text)
                or len(new_text) < len(old_text)
            ):
                continue

            similarity = _edit_similarity(old_text, new_text)
            reliable_ocr = (
                float(old_conf) >= .45
                or float(new_conf) >= .45
            )

            if not reliable_ocr:
                # Low-confidence OCR is not a trustworthy geometry veto.
                # In this regime the temporal glyph consensus is the primary
                # signal; OCR only checks that the candidate did not collapse
                # or lose a substantial part of the line.
                preserved = similarity >= max(.60, float(min_similarity))
            elif edge == 0:
                preserved = (
                    new_text[0] == old_text[0]
                    and similarity >= float(min_similarity)
                )
            elif edge == 2:
                preserved = (
                    new_text[-1] == old_text[-1]
                    and similarity >= float(min_similarity)
                )
            else:
                # Vertical clipping can mutate any glyph in the line, so use
                # a stricter whole-string guard plus both boundary characters.
                preserved = (
                    new_text[0] == old_text[0]
                    and new_text[-1] == old_text[-1]
                    and similarity >= max(.90, float(min_similarity))
                )
            if preserved:
                passed += 1
        return checks > 0 and passed * 2 >= checks

    out = anchor.copy()
    # Horizontal edges first. If the most aggressive candidate is unsafe,
    # search between the safe anchor and the candidate for the tightest
    # pixel position that still preserves OCR content.
    for edge in (0, 2, 1, 3):
        base_value = float(out[edge])
        target_value = float(candidate[edge])
        if edge in (0, 1):
            inward = target_value > base_value + 0.5
        else:
            inward = target_value < base_value - 0.5
        if not inward:
            continue

        trial = out.copy()
        trial[edge] = target_value
        if trial[2] <= trial[0] or trial[3] <= trial[1]:
            continue
        if trial_is_safe(trial, edge):
            out = trial
            continue

        safe_value = int(round(base_value))
        unsafe_value = int(round(target_value))
        if edge in (0, 1):
            while unsafe_value - safe_value > 1:
                mid = (safe_value + unsafe_value) // 2
                trial = out.copy()
                trial[edge] = float(mid)
                if trial_is_safe(trial, edge):
                    safe_value = mid
                else:
                    unsafe_value = mid
        else:
            while safe_value - unsafe_value > 1:
                mid = (safe_value + unsafe_value) // 2
                trial = out.copy()
                trial[edge] = float(mid)
                if trial_is_safe(trial, edge):
                    safe_value = mid
                else:
                    unsafe_value = mid

        if float(safe_value) != float(out[edge]):
            trial = out.copy()
            trial[edge] = float(safe_value)
            if trial[2] > trial[0] and trial[3] > trial[1]:
                out = trial

    return out


def _glyph_candidate_preserves_text(
    frame_map,
    sample_ids,
    anchor,
    candidate,
    recognizer,
    *,
    min_similarity=.65,
):
    if recognizer is None:
        return False
    checks = 0
    passed = 0
    ids = _sample_ids(sample_ids, min(3, len(sample_ids)))
    for fi in ids:
        frame = frame_map.get(fi)
        if frame is None:
            continue
        h, w = frame.shape[:2]

        def crop(box):
            x1 = max(0, int(np.floor(float(box[0]))))
            y1 = max(0, int(np.floor(float(box[1]))))
            x2 = min(w, int(np.ceil(float(box[2]))))
            y2 = min(h, int(np.ceil(float(box[3]))))
            return frame[y1:y2, x1:x2]

        old_text, _ = recognizer.recognize(crop(anchor))
        new_text, _ = recognizer.recognize(crop(candidate))
        old_text = _normalized_text(old_text)
        new_text = _normalized_text(new_text)
        if not old_text:
            continue
        checks += 1
        if not new_text:
            continue
        preserves_left = (
            float(candidate[0]) <= float(anchor[0]) + 0.5
            or new_text[0] == old_text[0]
        )
        preserves_right = (
            float(candidate[2]) >= float(anchor[2]) - 0.5
            or new_text[-1] == old_text[-1]
        )
        if (
            new_text
            and contains_cjk(new_text)
            and len(new_text) >= len(old_text)
            and preserves_left
            and preserves_right
            and _edit_similarity(old_text, new_text) >= float(min_similarity)
        ):
            passed += 1
    return checks > 0 and passed * 2 >= checks


def tighten_v1_static_tracks_with_temporal_glyphs(
    video_path,
    tracks,
    identity,
    recognizer,
    *,
    sample_count=15,
    safety_pad=4,
):
    tracks = list(tracks)
    identity = dict(identity)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    tightened = 0
    ratios = []
    for track in tracks:
        fs = track.sorted_frames()
        if not fs or _track_edge_range(track) > 1e-6:
            continue
        anchor = track.observations[fs[0]].bbox.copy()
        sample_ids = _sample_ids(fs, int(sample_count))
        frame_map = _read_frames(video_path, sample_ids)
        frames = [frame_map[fi] for fi in sample_ids if fi in frame_map]
        candidate = _temporal_glyph_candidate(
            frames,
            anchor,
            safety_pad=safety_pad,
        )
        candidate = _guard_temporal_candidate_edges(
            frame_map,
            sample_ids,
            anchor,
            candidate,
            recognizer,
        )
        old_area = max(
            1.0,
            float((anchor[2] - anchor[0]) * (anchor[3] - anchor[1])),
        )
        new_area = float(
            (candidate[2] - candidate[0])
            * (candidate[3] - candidate[1])
        )
        if new_area >= .98 * old_area:
            continue
        for fi in fs:
            track.observations[fi].bbox = candidate.copy()
        tightened += 1
        ratios.append(new_area / old_area)

    # Tightening is independent per line. Restore the strong centered-layout
    # prior after shrinkage without forcing sibling widths to match.
    sid_members = {}
    by_id = {t.track_id: t for t in tracks}
    for tid, (sid, line_id) in identity.items():
        sid_members.setdefault(int(sid), []).append((int(line_id), int(tid)))
    center_adjustments = 0
    for members in sid_members.values():
        if len(members) < 2:
            continue
        ordered = [
            by_id[tid]
            for _, tid in sorted(members)
            if tid in by_id
        ]
        if len(ordered) < 2:
            continue
        common = set(ordered[0].sorted_frames())
        for track in ordered[1:]:
            common &= set(track.sorted_frames())
        common = sorted(common)
        center_adjustments += _align_static_sibling_centers(
            ordered,
            common,
            frame_width=frame_width,
            max_shift_px=4.0,
        )

    return {
        "glyph_tightened_track_count": int(tightened),
        "glyph_tightening_area_ratio_median": (
            float(np.median(np.asarray(ratios, np.float32)))
            if ratios else 1.0
        ),
        "glyph_center_alignment_adjustment_count": int(center_adjustments),
    }


def merge_v1_same_content_tracks(
    video_path,
    tracks,
    identity,
    recognizer=None,
    *,
    text_evidence=None,
    visual_evidence=None,
    max_gap=1,
):
    tracks = list(tracks)
    identity = dict(identity)
    by_id = {t.track_id: t for t in tracks}

    sid_members = {}
    for tid, (sid, line_id) in identity.items():
        sid_members.setdefault(int(sid), []).append((int(line_id), int(tid)))

    eligible = []
    for track in tracks:
        tag = identity.get(track.track_id)
        if tag is None:
            continue
        sid, line_id = tag
        members = sid_members.get(int(sid), [])
        if int(line_id) != 0 or len(members) != 1:
            continue
        fs = track.sorted_frames()
        if fs:
            eligible.append((fs[0], fs[-1], track.track_id))
    eligible.sort()

    if text_evidence is None:
        text_evidence = _collect_track_text_evidence(
            video_path,
            [by_id[tid] for _, _, tid in eligible],
            recognizer,
        )
    else:
        text_evidence = {
            int(k): list(v)
            for k, v in dict(text_evidence).items()
        }

    visual_evidence = dict(visual_evidence or {})
    parent = {tid: tid for _, _, tid in eligible}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pair_metrics = []
    for left, right in zip(eligible, eligible[1:]):
        a_start, a_end, a_id = left
        b_start, b_end, b_id = right
        gap = int(b_start) - int(a_end) - 1
        if gap < 0 or gap > int(max_gap):
            continue

        texts_a = text_evidence.get(a_id, [])
        texts_b = text_evidence.get(b_id, [])
        text_sim = _max_text_similarity(texts_a, texts_b)
        key = (a_id, b_id)
        if key in visual_evidence:
            visual_sim = float(visual_evidence[key])
        elif video_path is not None:
            visual_sim = _boundary_visual_similarity(
                video_path,
                by_id[a_id],
                by_id[b_id],
            )
            visual_evidence[key] = visual_sim
        else:
            visual_sim = 0.0

        len_a = int(a_end - a_start + 1)
        len_b = int(b_end - b_start + 1)
        strong_text_match = text_sim >= .65
        short_fragment_match = (
            text_sim >= .35
            and visual_sim >= .70
            and min(len_a, len_b) <= 5
        )
        no_text_evidence=not texts_a and not texts_b
        box_a=np.median(
            np.stack([o.bbox for o in by_id[a_id].observations.values()]),
            axis=0,
        )
        box_b=np.median(
            np.stack([o.bbox for o in by_id[b_id].observations.values()]),
            axis=0,
        )
        wa=max(1.0,float(box_a[2]-box_a[0]))
        ha=max(1.0,float(box_a[3]-box_a[1]))
        wb=max(1.0,float(box_b[2]-box_b[0]))
        hb=max(1.0,float(box_b[3]-box_b[1]))
        width_ratio=wa/wb
        height_ratio=ha/hb
        visual_only_match=(
            no_text_evidence
            and gap<=1
            and visual_sim>=.94
            and .85<=width_ratio<=1.18
            and .85<=height_ratio<=1.18
        )
        should_merge = (
            (
                visual_sim >= .65
                and (strong_text_match or short_fragment_match)
            )
            or visual_only_match
        )
        pair_metrics.append(
            {
                "left_track_id": a_id,
                "right_track_id": b_id,
                "text_similarity": float(text_sim),
                "visual_similarity": float(visual_sim),
                "merged": bool(should_merge),
            }
        )
        if should_merge:
            union(a_id, b_id)

    groups = {}
    for _, _, tid in eligible:
        groups.setdefault(find(tid), []).append(tid)
    groups = [
        sorted(
            tids,
            key=lambda tid: by_id[tid].sorted_frames()[0],
        )
        for tids in groups.values()
        if len(tids) > 1
    ]

    removed = set()
    compact_candidates = {}
    for tids in groups:
        base_id = tids[0]
        base = by_id[base_id]
        compact_candidates[base_id] = [
            _track_median_bbox(by_id[tid])
            for tid in tids
        ]
        for tid in tids[1:]:
            other = by_id[tid]
            for fi, obs in other.observations.items():
                base.observations[int(fi)] = obs
            if (
                base.language_status != "validated_chinese"
                and other.language_status == "validated_chinese"
            ):
                base.language_status = other.language_status
            removed.add(tid)
            identity.pop(tid, None)

    out = [t for t in tracks if t.track_id not in removed]
    out.sort(
        key=lambda t: (
            min(t.sorted_frames()) if t.sorted_frames() else 10**12,
            t.track_id,
        )
    )
    return (
        out,
        identity,
        compact_candidates,
        {
            "merged_group_count": len(groups),
            "merged_track_count": len(removed),
            "merge_pair_metrics": pair_metrics,
        },
    )


def _compact_merged_bbox(candidates, frame_width):
    arr = np.stack(
        [np.asarray(b, dtype=np.float32) for b in candidates],
        axis=0,
    )
    areas = (
        np.maximum(0.0, arr[:, 2] - arr[:, 0])
        * np.maximum(0.0, arr[:, 3] - arr[:, 1])
    )
    box = arr[int(np.argmin(areas))].copy()
    width = float(box[2] - box[0])
    cx = .5 * float(box[0] + box[2])
    screen_cx = .5 * float(frame_width)
    if 4.0 < abs(cx - screen_cx) <= 20.0:
        box[0] = screen_cx - width * .5
        box[2] = screen_cx + width * .5
    box[0] = max(0.0, min(float(frame_width) - width, float(box[0])))
    box[2] = box[0] + width
    return box.astype(np.float32)


def _expanded_ocr_candidate(frame, anchor, recognizer):
    if recognizer is None or not hasattr(recognizer, "reader"):
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(v))) for v in anchor]
    if x2 <= x1 or y2 <= y1:
        return None
    bw = x2 - x1
    bh = y2 - y1
    pad_x = max(20, int(round(.08 * bw)))
    pad_y = max(14, int(round(.28 * bh)))
    ex1 = max(0, x1 - pad_x)
    ey1 = max(0, y1 - pad_y)
    ex2 = min(w, x2 + pad_x)
    ey2 = min(h, y2 + pad_y)
    rows = recognizer.reader.readtext(
        frame[ey1:ey2, ex1:ex2],
        detail=1,
        paragraph=False,
    )
    candidates = []
    for row in rows:
        if len(row) < 3:
            continue
        poly, text, conf = row[0], str(row[1]), float(row[2])
        if not poly:
            continue
        xs = [float(pt[0]) + ex1 for pt in poly]
        ys = [float(pt[1]) + ey1 for pt in poly]
        box = np.asarray(
            [min(xs), min(ys), max(xs), max(ys)],
            dtype=np.float32,
        )
        cx = .5 * float(box[0] + box[2])
        cy = .5 * float(box[1] + box[3])
        if not (
            x1 - 10 <= cx <= x2 + 10
            and y1 - 10 <= cy <= y2 + 10
        ):
            continue
        candidates.append((box, text, conf))
    if not candidates:
        return None
    box = np.asarray(
        [
            min(float(row[0][0]) for row in candidates),
            min(float(row[0][1]) for row in candidates),
            max(float(row[0][2]) for row in candidates),
            max(float(row[0][3]) for row in candidates),
        ],
        dtype=np.float32,
    )
    text = "".join(row[1] for row in candidates)
    conf = float(np.mean([row[2] for row in candidates]))
    return box, text, conf


def tighten_v1_static_tracks_with_ocr(
    video_path,
    tracks,
    identity,
    recognizer,
    *,
    min_height_px=58,
    sample_count=3,
    safety_pad=2,
):
    tracks = list(tracks)
    identity = dict(identity)
    if recognizer is None or not hasattr(recognizer, "reader"):
        return {
            "ocr_tightened_track_count": 0,
            "ocr_tightening_area_ratio_median": 1.0,
        }

    sid_members = {}
    for tid, (sid, line_id) in identity.items():
        sid_members.setdefault(int(sid), []).append((int(line_id), int(tid)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")

    tightened = 0
    area_ratios = []
    try:
        for track in tracks:
            tag = identity.get(track.track_id)
            if tag is None:
                continue
            sid, line_id = tag
            if int(line_id) != 0 or len(sid_members.get(int(sid), [])) != 1:
                continue
            fs = track.sorted_frames()
            if not fs or _track_edge_range(track) > 1e-6:
                continue
            anchor = track.observations[fs[0]].bbox.copy()
            height = float(anchor[3] - anchor[1])
            if height < float(min_height_px):
                continue

            if len(fs) <= int(sample_count):
                sample_ids = list(fs)
            elif len(fs) <= 5:
                sample_ids = _sample_ids(fs, int(sample_count))
            else:
                positions = np.linspace(.25, .75, int(sample_count))
                sample_ids = sorted({
                    fs[int(round((len(fs) - 1) * float(p)))]
                    for p in positions
                })

            proposals = []
            for fi in sample_ids:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
                ok, frame = cap.read()
                if not ok:
                    continue
                h, w = frame.shape[:2]
                x1 = max(0, int(np.floor(float(anchor[0]))))
                y1 = max(0, int(np.floor(float(anchor[1]))))
                x2 = min(w, int(np.ceil(float(anchor[2]))))
                y2 = min(h, int(np.ceil(float(anchor[3]))))
                if x2 <= x1 or y2 <= y1:
                    continue
                current_text, _ = recognizer.recognize(
                    frame[y1:y2, x1:x2]
                )
                expanded = _expanded_ocr_candidate(
                    frame,
                    anchor,
                    recognizer,
                )
                if expanded is None:
                    continue
                candidate, candidate_text, _ = expanded
                proposal = guarded_ocr_tighten_bbox(
                    anchor,
                    current_text,
                    candidate,
                    candidate_text,
                    safety_pad=safety_pad,
                )
                if not np.allclose(proposal, anchor):
                    proposals.append(proposal)

            if not proposals:
                continue
            stack = np.stack(proposals, axis=0)
            final_box = np.asarray(
                [
                    np.min(stack[:, 0]),
                    np.min(stack[:, 1]),
                    np.max(stack[:, 2]),
                    np.max(stack[:, 3]),
                ],
                dtype=np.float32,
            )
            old_area = max(
                1.0,
                float(
                    (anchor[2] - anchor[0])
                    * (anchor[3] - anchor[1])
                ),
            )
            new_area = float(
                (final_box[2] - final_box[0])
                * (final_box[3] - final_box[1])
            )
            if new_area >= .98 * old_area:
                continue
            for fi in fs:
                track.observations[fi].bbox = final_box.copy()
            tightened += 1
            area_ratios.append(new_area / old_area)
    finally:
        cap.release()

    return {
        "ocr_tightened_track_count": int(tightened),
        "ocr_tightening_area_ratio_median": (
            float(np.median(np.asarray(area_ratios, np.float32)))
            if area_ratios else 1.0
        ),
    }


def regularize_v1_single_line_height(
    tracks,
    identity,
    *,
    frame_height,
    min_reference_tracks=8,
    center_window_px=16.0,
    safety_extra_px=4.0,
    outlier_extra_px=8.0,
    outlier_ratio=1.15,
):
    """Regularize only implausibly tall single-line boxes.

    The hard-subtitle video uses a stable font size, but detector noise can
    leave a small number of lines vertically inflated. The reference height
    is learned from neighboring single-line tracks with a similar vertical
    center, so this does not force multiline roles or other screen bands into
    one global geometry.
    """
    tracks = list(tracks)
    identity = dict(identity)
    sid_members = {}
    for tid, (sid, line_id) in identity.items():
        sid_members.setdefault(int(sid), []).append((int(line_id), int(tid)))

    entries = []
    for track in tracks:
        tag = identity.get(track.track_id)
        if tag is None:
            continue
        sid, line_id = tag
        if int(line_id) != 0 or len(sid_members.get(int(sid), [])) != 1:
            continue
        fs = track.sorted_frames()
        if not fs or _track_edge_range(track) > 1e-6:
            continue
        box = track.observations[fs[0]].bbox
        height = float(box[3] - box[1])
        cy = .5 * float(box[1] + box[3])
        if height <= 0:
            continue
        entries.append((track, height, cy))

    adjusted = 0
    before_heights = []
    after_heights = []
    reference_medians = []

    for track, height, cy in entries:
        local = [
            h
            for _, h, ref_cy in entries
            if abs(float(ref_cy) - float(cy)) <= float(center_window_px)
        ]
        if len(local) < int(min_reference_tracks):
            continue
        median_h = float(np.median(np.asarray(local, dtype=np.float32)))
        threshold = max(
            median_h + float(outlier_extra_px),
            median_h * float(outlier_ratio),
        )
        if height <= threshold:
            continue

        target_h = min(
            height,
            median_h + float(safety_extra_px),
        )
        # Never allow this regularizer to remove more than ~25% of a line's
        # height in one step. Extreme transition artifacts are handled by the
        # transition-fragment stage, not by font-size normalization.
        target_h = max(target_h, .75 * height)
        if target_h >= height - 0.5:
            continue

        fs = track.sorted_frames()
        first = track.observations[fs[0]].bbox
        center_y = .5 * float(first[1] + first[3])
        y1 = center_y - .5 * target_h
        y2 = center_y + .5 * target_h
        if y1 < 0:
            y2 -= y1
            y1 = 0.0
        if y2 > float(frame_height):
            y1 -= y2 - float(frame_height)
            y2 = float(frame_height)
        y1 = max(0.0, y1)
        y2 = min(float(frame_height), y2)
        if y2 <= y1:
            continue

        before_heights.append(height)
        for fi in fs:
            box = track.observations[fi].bbox.copy()
            box[1] = float(y1)
            box[3] = float(y2)
            track.observations[fi].bbox = box
        after_heights.append(float(y2 - y1))
        reference_medians.append(median_h)
        adjusted += 1

    return {
        "height_regularized_track_count": int(adjusted),
        "height_regularizer_reference_median_px": (
            float(np.median(np.asarray(reference_medians, np.float32)))
            if reference_medians else 0.0
        ),
        "height_regularizer_before_median_px": (
            float(np.median(np.asarray(before_heights, np.float32)))
            if before_heights else 0.0
        ),
        "height_regularizer_after_median_px": (
            float(np.median(np.asarray(after_heights, np.float32)))
            if after_heights else 0.0
        ),
    }


def apply_v1_static_geometry_lock(
    video_path,
    tracks,
    identity,
    *,
    sample_count=15,
    pad_px=2,
    compact_candidates=None,
):
    tracks = list(tracks)
    identity = dict(identity)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    compact_candidates = dict(compact_candidates or {})
    pre_ranges = [_track_edge_range(t) for t in tracks]
    static_ids = set()
    moving_ids = set()
    merged_compact_ids = set()

    for track in tracks:
        fs = track.sorted_frames()
        if not fs:
            continue
        arr = np.stack(
            [track.observations[fi].bbox for fi in fs],
            axis=0,
        ).astype(np.float32)

        if detect_consistent_translation(arr):
            _lock_moving_track(track)
            moving_ids.add(track.track_id)
        else:
            _lock_static_track(
                video_path,
                track,
                sample_count=sample_count,
                pad_px=pad_px,
            )
            # Do not override temporal consensus with the smallest member
            # box from a stitched group. The smallest detector box is often
            # the clipped one; logical stitching should widen the evidence
            # window, not select the most aggressive member geometry.
            static_ids.add(track.track_id)

    by_subtitle = {}
    for track in tracks:
        tag = identity.get(track.track_id)
        if tag is None:
            continue
        subtitle_id, line_id = tag
        by_subtitle.setdefault(int(subtitle_id), []).append(
            (int(line_id), track)
        )

    seam_adjustments = 0
    center_align_adjustments = 0
    center_offsets = []
    for _, group in sorted(by_subtitle.items()):
        if len(group) < 2:
            continue
        ordered = [
            track
            for _, track in sorted(
                group,
                key=lambda item: (item[0], item[1].track_id),
            )
        ]
        common = set(ordered[0].sorted_frames())
        for track in ordered[1:]:
            common &= set(track.sorted_frames())
        common = sorted(common)

        # A true moving line is allowed to translate. For the hard-subtitle
        # case both siblings are static, so use one temporal seam for the
        # whole subtitle lifecycle.
        if all(t.track_id in static_ids for t in ordered):
            center_align_adjustments += _align_static_sibling_centers(
                ordered,
                common,
                frame_width=frame_width,
                max_shift_px=4.0,
            )
            seam_adjustments += _fixed_multiline_seam(
                video_path,
                ordered,
                common,
                sample_count=sample_count,
            )

        for fi in common:
            for a, b in zip(ordered, ordered[1:]):
                ba = a.observations[fi].bbox
                bb = b.observations[fi].bbox
                cxa = .5 * float(ba[0] + ba[2])
                cxb = .5 * float(bb[0] + bb[2])
                center_offsets.append(abs(cxa - cxb))

    post_ranges = [_track_edge_range(t) for t in tracks]
    areas = []
    final_overlap_frames = 0
    final_overlap_pixels = 0.0
    frame_groups = {}
    for track in tracks:
        tag = identity.get(track.track_id)
        if tag is None:
            continue
        subtitle_id, line_id = tag
        for fi in track.sorted_frames():
            box = track.observations[fi].bbox
            areas.append(
                max(0.0, float(box[2] - box[0]))
                * max(0.0, float(box[3] - box[1]))
            )
            frame_groups.setdefault(
                (int(fi), int(subtitle_id)),
                [],
            ).append((int(line_id), box))

    for rows in frame_groups.values():
        if len(rows) < 2:
            continue
        rows = sorted(rows, key=lambda item: item[0])
        overlap = sum(
            line_overlap_area(a[1], b[1])
            for a, b in zip(rows, rows[1:])
        )
        final_overlap_pixels += overlap
        if overlap > 0:
            final_overlap_frames += 1

    def pct(values, q):
        if not values:
            return 0.0
        return float(np.percentile(np.asarray(values, np.float32), q))

    return {
        "static_locked_track_count": len(static_ids),
        "motion_compensated_track_count": len(moving_ids),
        "merged_compact_track_count": len(merged_compact_ids),
        "prelock_edge_range_median_px": pct(pre_ranges, 50),
        "prelock_edge_range_p95_px": pct(pre_ranges, 95),
        "prelock_max_edge_range_px": max(pre_ranges, default=0.0),
        "postlock_edge_range_median_px": pct(post_ranges, 50),
        "postlock_edge_range_p95_px": pct(post_ranges, 95),
        "postlock_max_edge_range_px": max(post_ranges, default=0.0),
        "canonical_bbox_area_median": pct(areas, 50),
        "multiline_center_offset_median_px": pct(center_offsets, 50),
        "multiline_center_offset_p95_px": pct(center_offsets, 95),
        "center_alignment_adjustment_count": int(center_align_adjustments),
        "temporal_seam_adjustment_count": int(seam_adjustments),
        "final_overlap_frame_count": int(final_overlap_frames),
        "final_overlap_pixel_count": int(round(final_overlap_pixels)),
    }
