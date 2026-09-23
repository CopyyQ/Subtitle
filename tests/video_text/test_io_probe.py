from pathlib import Path
import pytest
from src.video_text.io_probe import bottom_roi, probe_video

def test_bottom_45_percent_reaches_bottom_edge():
    y1, y2 = bottom_roi(1080, 0.45)
    assert y1 == 594
    assert y2 == 1080

@pytest.mark.parametrize("fraction", [0.25, 0.35, 0.45])
def test_allowed_roi_fractions(fraction):
    y1, y2 = bottom_roi(720, fraction)
    assert 0 <= y1 < y2 == 720

@pytest.mark.parametrize("fraction", [0.24, 0.46])
def test_roi_outside_assignment_range_is_rejected(fraction):
    with pytest.raises(ValueError, match="0.25.*0.45"):
        bottom_roi(720, fraction)

def test_supplied_video_probe():
    p = Path(__file__).resolve().parents[2] / "AI Engineer test.mp4"
    if not p.exists():
        pytest.skip("private evaluation video is not included in the Git repository")
    info = probe_video(p)
    assert info.width == 720
    assert info.height == 1280
    assert info.frame_count == 3733
    assert info.fps == 30.0
