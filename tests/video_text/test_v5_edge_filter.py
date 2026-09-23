import numpy as np

from src.video_text.v5_extent import _glyph_like_run, _recover_occluded_edges


def test_tiny_enhanced_side_speck_is_not_a_glyph():
    h=64
    assert not _glyph_like_run(width=7,area=88,h=h,yspan=18)


def test_normal_chinese_glyph_is_glyph_like():
    h=64
    assert _glyph_like_run(width=44,area=700,h=h,yspan=44)


def test_thin_horizontal_chinese_leader_is_glyph_like():
    h=42
    assert _glyph_like_run(width=42,area=110,h=h,yspan=5)


def test_edge_recovery_refuses_large_background_extension():
    # Stable core begins at x=88. A background component repeatedly reaches
    # x=42: that is >0.4 font-heights outside the core and must be ignored.
    h=70
    masks=[]
    for _ in range(3):
        m=np.zeros((70,200),np.uint8)
        m[10:60,42:132]=1
        masks.append(m)
    left,right=_recover_occluded_edges(
        masks,left=88,right=160,h=h,max_gap=.45*h
    )
    assert left==88


def test_edge_recovery_allows_small_occlusion_extension():
    h=64
    masks=[]
    for _ in range(3):
        m=np.zeros((64,220),np.uint8)
        m[8:58,153:210]=1
        masks.append(m)
    left,right=_recover_occluded_edges(
        masks,left=162,right=210,h=h,max_gap=.45*h
    )
    assert left<=153
