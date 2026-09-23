import pytest
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/"scripts/run_subtitle_pipeline.py"
VIDEO=ROOT/"AI Engineer test.mp4"

def run(*args):
    return subprocess.run([sys.executable,str(SCRIPT),*map(str,args)],
                          cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)

def test_dry_probe_reports_video_without_loading_detector():
    if not VIDEO.exists():
        pytest.skip("private evaluation video is not included in the Git repository")
    p=run(VIDEO,"--dry-probe")
    assert p.returncode==0
    assert '"width": 720' in p.stdout
    assert '"height": 1280' in p.stdout

def test_roi_outside_assignment_range_is_rejected():
    p=run(VIDEO,"--dry-probe","--roi-bottom-fraction","0.46")
    assert p.returncode!=0
    assert "0.25" in p.stderr and "0.45" in p.stderr

def test_unsupported_codec_is_rejected():
    p=run(VIDEO,"--dry-probe","--codec","vp9")
    assert p.returncode!=0

def test_export_srt_requires_recognition_mode():
    p=run(VIDEO,"--output","x.mp4","--export-srt","x.srt")
    assert p.returncode!=0
    assert "recognize" in p.stderr.lower()

def test_export_srt_with_recognition_is_not_silent_noop():
    p=run(VIDEO,"--dry-probe","--recognize-text","--export-srt","x.srt")
    assert p.returncode!=0
    assert "coordinate" in p.stderr.lower()


def test_cli_accepts_v55_temporal_mode_in_dry_probe():
    if not VIDEO.exists():
        pytest.skip("private evaluation video is not included in the Git repository")
    p=run(VIDEO,"--dry-probe","--temporal-mode","v5_5")
    assert p.returncode==0


def test_cli_accepts_v1_temporal_mode_in_dry_probe():
    if not VIDEO.exists():
        pytest.skip("private evaluation video is not included in the Git repository")
    p=run(VIDEO,"--dry-probe","--temporal-mode","v1")
    assert p.returncode==0
