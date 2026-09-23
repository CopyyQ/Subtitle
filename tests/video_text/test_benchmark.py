from pathlib import Path

from src.video_text.benchmark import (
    collect_environment_metadata,
    latency_summary,
    real_time_factor,
)


def test_latency_summary_reports_mean_p50_p95():
    s=latency_summary([1.0,2.0,3.0,100.0])
    assert s["mean_ms"]==26.5
    assert 2.0<=s["p50_ms"]<=3.0
    assert s["p95_ms"]>3.0


def test_real_time_factor():
    assert real_time_factor(50.0,100.0)==0.5


def test_environment_metadata_hashes_checkpoint_without_network(tmp_path):
    ckpt=tmp_path/"model.pth"
    ckpt.write_bytes(b"abc")
    data=collect_environment_metadata(Path(tmp_path),ckpt)
    assert data["checkpoint_name"]=="model.pth"
    assert data["checkpoint_sha256"]=="ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert data["python_version"]
    assert data["torch_version"]
