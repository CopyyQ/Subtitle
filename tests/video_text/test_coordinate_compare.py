import json

from scripts.compare_coordinates import compare_payloads, select_best_variant


def payload(records, **metrics):
    return {"metadata":metrics, "records":records}


def row(frame, box, subtitle_id=1, line_id=0):
    return {
        "frame":frame,
        "subtitle_id":subtitle_id,
        "line_id":line_id,
        "track_id":subtitle_id,
        "bbox":list(box),
    }


def test_identical_coordinates_pass():
    g=payload([row(1,[10,20,100,60])],multiline_overlap_frame_count=0)
    result=compare_payloads(g,g,max_edge_delta=2,max_missing_frame_rate=0.0)
    assert result["passed"] is True
    assert result["missing_key_count"]==0
    assert result["extra_key_count"]==0
    assert result["max_edge_delta_px"]==0.0


def test_one_pixel_edge_difference_passes_two_pixel_tolerance():
    g=payload([row(1,[10,20,100,60])])
    c=payload([row(1,[11,20,99,61])])
    result=compare_payloads(g,c,max_edge_delta=2,max_missing_frame_rate=0.0)
    assert result["passed"] is True
    assert result["max_edge_delta_px"]==1.0


def test_missing_subtitle_frame_fails():
    g=payload([row(1,[10,20,100,60]),row(2,[10,20,100,60])])
    c=payload([row(1,[10,20,100,60])])
    result=compare_payloads(g,c,max_edge_delta=2,max_missing_frame_rate=0.0)
    assert result["passed"] is False
    assert result["missing_key_count"]==1
    assert result["missing_boxed_frames"]==[2]


def test_added_ghost_frame_fails():
    g=payload([row(1,[10,20,100,60])])
    c=payload([row(1,[10,20,100,60]),row(2,[10,20,100,60])])
    result=compare_payloads(g,c,max_edge_delta=2,max_missing_frame_rate=0.0)
    assert result["passed"] is False
    assert result["extra_key_count"]==1
    assert result["ghost_boxed_frames"]==[2]


def test_multiline_overlap_regression_fails():
    g=payload([row(1,[10,20,100,60])],multiline_overlap_frame_count=0)
    c=payload([row(1,[10,20,100,60])],multiline_overlap_frame_count=1)
    result=compare_payloads(g,c,max_edge_delta=2,max_missing_frame_rate=0.0)
    assert result["passed"] is False
    assert "multiline_overlap_frame_count" in result["quality_regressions"]


def test_selection_uses_fastest_passing_then_safer_tie():
    rows=[
        {"status":"ok","comparison_passed":True,"end_to_end_fps":30.0,"batch_size":16,"precision":"fp16"},
        {"status":"ok","comparison_passed":True,"end_to_end_fps":29.5,"batch_size":8,"precision":"fp32"},
        {"status":"ok","comparison_passed":False,"end_to_end_fps":40.0,"batch_size":4,"precision":"fp16"},
    ]
    best=select_best_variant(rows)
    assert best["batch_size"]==8
    assert best["precision"]=="fp32"
