import numpy as np
from src.video_text.types import Candidate
from src.video_text.line_grouping import group_candidates_to_lines

def box(x1,y1,x2,y2,score=.9,level="HIGH"):
    return Candidate(np.array([x1,y1,x2,y2], np.float32), score, level)

def test_characters_on_same_baseline_merge_to_one_line():
    out=group_candidates_to_lines([
        box(100,800,150,840),
        box(158,801,210,841),
    ])
    assert len(out)==1
    assert out[0].bbox[0] <= 100
    assert out[0].bbox[2] >= 210

def test_two_subtitle_lines_stay_separate():
    out=group_candidates_to_lines([
        box(100,780,250,810),
        box(110,825,260,855),
    ])
    assert len(out)==2

def test_high_component_promotes_merged_line_to_high():
    out=group_candidates_to_lines([
        box(100,800,150,840,.78,"LOW"),
        box(158,801,210,841,.93,"HIGH"),
    ])
    assert len(out)==1
    assert out[0].level=="HIGH"
    assert out[0].score==.93
