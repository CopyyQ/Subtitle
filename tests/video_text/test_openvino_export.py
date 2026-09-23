from pathlib import Path

import numpy as np

from scripts.export_ppocr_openvino import export_fp32_ir


class FakeOV:
    def __init__(self):
        self.converted = []
        self.saved = []

    def convert_model(self, path):
        self.converted.append(str(path))
        return "MODEL"

    def save_model(self, model, path, compress_to_fp16=False):
        self.saved.append((model, str(path), compress_to_fp16))


def test_export_fp32_ir_converts_onnx_and_disables_fp16_compression(tmp_path):
    onnx = tmp_path / "inference.onnx"
    onnx.write_bytes(b"fake")
    out = tmp_path / "openvino" / "model.xml"
    ov = FakeOV()

    result = export_fp32_ir(onnx, out, ov_module=ov)

    assert result == out
    assert ov.converted == [str(onnx)]
    assert ov.saved == [("MODEL", str(out), False)]


def test_calibration_indices_include_forced_frames_without_exceeding_budget():
    from scripts.export_ppocr_openvino import build_calibration_frame_indices

    ids = build_calibration_frame_indices(
        total_frames=1000,
        max_samples=10,
        forced_frames=[10, 500, 999],
    )

    assert len(ids) == 10
    assert ids == sorted(set(ids))
    assert {10, 500, 999} <= set(ids)


class FakeNNCF:
    class TargetDevice:
        CPU = "CPU"

    class Dataset:
        def __init__(self, data):
            self.data = list(data)

    def __init__(self):
        self.calls = []

    def quantize(self, model, dataset, *, subset_size, target_device):
        self.calls.append((model, dataset.data, subset_size, target_device))
        return "INT8_MODEL"


class FakeOVQuant:
    class Core:
        def __init__(self):
            pass

        def read_model(self, path):
            return ("FP32_MODEL", str(path))

    def __init__(self):
        self.saved = []

    def save_model(self, model, path, compress_to_fp16=False):
        self.saved.append((model, str(path), compress_to_fp16))


def test_quantize_int8_ir_uses_calibration_dataset_and_saves_uncompressed(tmp_path):
    from scripts.export_ppocr_openvino import quantize_int8_ir

    xml = tmp_path / "fp32.xml"
    xml.write_text("<xml/>")
    out = tmp_path / "int8" / "model.xml"
    samples = [
        np.zeros((1, 3, 32, 32), dtype=np.float32),
        np.ones((1, 3, 32, 32), dtype=np.float32),
    ]
    nncf = FakeNNCF()
    ov = FakeOVQuant()

    result = quantize_int8_ir(
        xml,
        out,
        samples,
        nncf_module=nncf,
        ov_module=ov,
    )

    assert result == out
    assert nncf.calls[0][2] == 2
    assert nncf.calls[0][3] == "CPU"
    assert ov.saved == [("INT8_MODEL", str(out), False)]


def test_export_script_bootstraps_repo_root_for_direct_execution(tmp_path):
    import subprocess
    import sys
    import cv2

    root = Path(__file__).resolve().parents[2]
    script = root / "scripts" / "export_ppocr_openvino.py"
    video = tmp_path / "tiny.avi"

    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (64, 64),
    )
    writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    writer.release()

    code = f"""
import runpy, sys
root = {str(root)!r}
scripts = {str(root / "scripts")!r}
sys.path = [scripts] + [p for p in sys.path if p not in ("", root)]
ns = runpy.run_path({str(script)!r}, run_name="not_main")
samples = ns["collect_calibration_inputs"]({str(video)!r}, [0])
print(len(samples))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        text=True,
        capture_output=True,
    )

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "1"
