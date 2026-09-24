import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "benchmark_ppocr_performance_gate.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ppocr_perf_gate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_variant_matrix_covers_batch_thread_hpi_and_concurrency():
    mod = load_module()
    variants = mod.build_variants(
        batches=(8, 16, 24, 32),
        include_hpi=True,
        include_onnx=True,
        include_trt=True,
        include_dual=True,
    )
    names = {v.name for v in variants}
    assert {"static_b8", "static_b16", "static_b24", "static_b32"} <= names
    assert {"threaded_b16", "threaded_b24", "threaded_b32"} <= names
    assert "hpi_b16" in names
    assert "onnx_b16" in names
    assert "trt_fp16_b16" in names
    assert "dual_static_b16" in names


def test_select_best_requires_quality_gate_then_highest_wall_fps():
    mod = load_module()
    rows = [
        {"name": "fast_bad", "wall_fps": 90.0, "quality_pass": False},
        {"name": "good_slow", "wall_fps": 42.0, "quality_pass": True},
        {"name": "good_fast", "wall_fps": 47.5, "quality_pass": True},
    ]
    best = mod.select_best_variant(rows)
    assert best["name"] == "good_fast"


def test_center_quality_gate_counts_misses_and_ghosts():
    mod = load_module()
    reference = {
        10: [[100, 100, 200, 140]],
        11: [[100, 100, 200, 140]],
    }
    candidate = {
        10: [[102, 101, 198, 139]],
        12: [[100, 100, 200, 140]],
    }
    q = mod.center_quality(reference, candidate, threshold_px=25.0)
    assert q["matched"] == 1
    assert q["reference_boxes"] == 2
    assert q["candidate_boxes"] == 2
    assert q["missing_boxed_frames"] == 1
    assert q["ghost_boxed_frames"] == 1
    assert q["quality_pass"] is False
