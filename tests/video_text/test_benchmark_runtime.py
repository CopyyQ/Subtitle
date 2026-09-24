import importlib.util
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[2]/"scripts"/"benchmark_runtime.py"

def load_module():
    spec=importlib.util.spec_from_file_location("benchmark_runtime",SCRIPT)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def test_benchmark_defaults_match_ppocr_production():
    mod=load_module()
    args=mod.build_parser().parse_args(["input.mp4","--output-dir","out"])
    assert args.detector=="ppocrv5_mobile"
    assert args.batches==[32]
    assert args.precisions==["fp32"]
    assert args.decode_prefetch_batches==4
    assert args.high_score==.84
    assert args.low_score==.50
    assert args.ppocr_thresh==.30
    assert args.ppocr_box_thresh==.50

def test_generated_command_contains_detector_and_ppocr_runtime_flags(tmp_path):
    mod=load_module()
    args=mod.build_parser().parse_args(["input.mp4","--output-dir",str(tmp_path)])
    cmd=mod._command(
        Path("input.mp4"),Path("out.mp4"),Path("coords.json"),
        args,32,"fp32",
    )
    assert "--detector ppocrv5_mobile" in cmd
    assert "--batch-size 32" in cmd
    assert "--decode-prefetch-batches 4" in cmd
    assert "--ppocr-thresh 0.3" in cmd
    assert "--ppocr-box-thresh 0.5" in cmd
    assert "--high-score 0.84" in cmd
    assert "--low-score 0.5" in cmd
