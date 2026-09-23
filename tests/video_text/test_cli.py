import pytest
from pathlib import Path
import subprocess
import sys

from scripts.run_subtitle_pipeline import build_parser
from src.video_text.pipeline import PipelineConfig

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


def test_production_cli_defaults_to_detection_only_auto_runtime():
    args=build_parser().parse_args(["input.mp4"])
    assert args.device=="auto"
    assert args.precision=="auto"
    assert args.batch_size==32
    assert args.decode_prefetch_batches==4
    assert args.cpu_threads==0
    assert args.box_thickness==2
    assert args.detector=="ppocrv5_mobile"
    assert args.high_score==.84
    assert args.low_score==.50
    assert args.ppocr_thresh==.30
    assert args.ppocr_box_thresh==.50
    assert args.validate_chinese is False


@pytest.mark.parametrize("device",["auto","cpu","cuda"])
def test_cli_accepts_device_choices(device):
    args=build_parser().parse_args(["input.mp4","--device",device])
    assert args.device==device


@pytest.mark.parametrize("precision",["auto","fp32","fp16"])
def test_cli_accepts_precision_choices(precision):
    args=build_parser().parse_args(["input.mp4","--precision",precision])
    assert args.precision==precision


def test_validate_chinese_is_explicit_opt_in():
    args=build_parser().parse_args(["input.mp4","--validate-chinese"])
    assert args.validate_chinese is True


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"batch_size":0},"batch_size"),
        ({"cpu_threads":-1},"cpu_threads"),
        ({"box_thickness":0},"box_thickness"),
    ],
)
def test_pipeline_config_rejects_invalid_runtime_values(kwargs,match):
    with pytest.raises(ValueError,match=match):
        PipelineConfig(**kwargs)
