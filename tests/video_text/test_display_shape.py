import cv2
import numpy as np

from src.video_text.display_shape import (
    rectangle_union_polygons,
    build_display_polygons,
)


def area(poly):
    pts=np.asarray(poly,np.float32).reshape(-1,1,2)
    return abs(float(cv2.contourArea(pts)))


def test_overlapping_two_line_boxes_become_one_non_rectangular_polygon():
    top=[100,100,500,160]
    bottom=[180,150,420,215]
    polygons=rectangle_union_polygons([top,bottom])
    assert len(polygons)==1
    poly=polygons[0]
    assert len(poly)>4
    expected=(400*60)+(240*65)-(240*10)
    assert area(poly)==expected
    outer=(500-100)*(215-100)
    assert area(poly)<outer


def test_close_two_lines_are_connected_into_one_step_polygon():
    records=[
        {"track_id":1,"bbox":[100,100,500,150]},
        {"track_id":2,"bbox":[180,156,420,206]},
    ]
    shapes=build_display_polygons(records)
    assert len(shapes)==1
    assert shapes[0]["track_ids"]==[1,2]
    assert len(shapes[0]["polygon"])>4


def test_two_far_apart_lines_remain_separate_shapes():
    records=[
        {"track_id":1,"bbox":[100,100,500,150]},
        {"track_id":2,"bbox":[180,260,420,310]},
    ]
    shapes=build_display_polygons(records)
    assert len(shapes)==2


def test_single_line_remains_rectangle_polygon():
    shapes=build_display_polygons([
        {"track_id":7,"bbox":[100,100,500,150]},
    ])
    assert len(shapes)==1
    assert shapes[0]["track_ids"]==[7]
    assert len(shapes[0]["polygon"])==4

def test_draw_uses_step_polygon_not_outer_rectangle():
    from src.video_text.display_shape import draw_display_polygons
    frame=np.zeros((320,640,3),np.uint8)
    records=[
        {"track_id":1,"bbox":[100,100,500,150]},
        {"track_id":2,"bbox":[180,156,420,206]},
    ]
    draw_display_polygons(frame,records,color=(0,255,0),thickness=2)
    # Actual outline exists on the short second line.
    assert frame[180,180,1] > 0
    # A giant bounding rectangle would draw a vertical edge here at x=100;
    # the step-polygon must leave this location empty.
    assert frame[180,100,1] == 0

def test_display_shape_records_emit_one_polygon_for_two_lines():
    from src.video_text.display_shape import build_display_shape_records
    records=[
        {"frame":10,"track_id":1,"bbox":[100,100,500,150]},
        {"frame":10,"track_id":2,"bbox":[180,156,420,206]},
    ]
    out=build_display_shape_records(records,fps=25.0)
    assert len(out)==1
    assert out[0]["frame"]==10
    assert out[0]["timestamp"]==0.4
    assert out[0]["track_ids"]==[1,2]
    assert len(out[0]["polygon"])>4


def test_v55_line_renderer_draws_two_rectangles_without_joining_them():
    from src.video_text.display_shape import draw_line_rectangles
    frame=np.zeros((320,640,3),np.uint8)
    records=[
        {"subtitle_id":7,"line_id":0,"track_id":11,"bbox":[100,100,500,150]},
        {"subtitle_id":7,"line_id":1,"track_id":12,"bbox":[180,150,420,205]},
    ]
    draw_line_rectangles(frame,records,color=(0,255,0),thickness=2)
    assert frame[150,100,1] > 0
    assert frame[175,100,1] == 0
    assert frame[175,180,1] > 0


def test_v55_line_shape_records_emit_one_rectangle_per_line():
    from src.video_text.display_shape import build_line_shape_records
    records=[
        {"frame":10,"subtitle_id":7,"line_id":0,"track_id":11,"bbox":[100,100,500,150]},
        {"frame":10,"subtitle_id":7,"line_id":1,"track_id":12,"bbox":[180,150,420,205]},
    ]
    out=build_line_shape_records(records,fps=25.0)
    assert len(out)==2
    assert {(r["subtitle_id"],r["line_id"]) for r in out}=={(7,0),(7,1)}
    assert all(len(r["polygon"])==4 for r in out)
