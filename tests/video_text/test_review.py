import pytest
from pathlib import Path
import cv2
from src.video_text.lifecycle import TemporalEvent
from src.video_text.review import export_event_contact_sheet

def test_event_contact_sheet_contains_context_frames():
    root=Path(__file__).resolve().parents[2]
    if not (root/"AI Engineer test.mp4").exists():
        pytest.skip("private evaluation video is not included in the Git repository")
    outdir=root/"tests/video_text/.tmp_review"
    event=TemporalEvent("track_end",1,71,71)
    out=export_event_contact_sheet(root/"AI Engineer test.mp4",event,outdir,.45,context=2)
    img=cv2.imread(str(out))
    assert img is not None
    assert img.shape[0]==576
    assert img.shape[1]==720*5
    assert "track_end" in out.name
