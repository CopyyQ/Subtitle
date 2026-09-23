from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[2]
EXTERNAL_VIDEO = ROOT / "AI Engineer test.mp4"

_EXTERNAL_VIDEO_TESTS = {
    "tests/video_text/test_cli.py::test_dry_probe_reports_video_without_loading_detector",
    "tests/video_text/test_cli.py::test_cli_accepts_v55_temporal_mode_in_dry_probe",
    "tests/video_text/test_cli.py::test_cli_accepts_v1_temporal_mode_in_dry_probe",
    "tests/video_text/test_horizontal_recovery.py::test_real_frame_1648_recovers_leading_yi_glyph",
    "tests/video_text/test_io_probe.py::test_supplied_video_probe",
    "tests/video_text/test_review.py::test_event_contact_sheet_contains_context_frames",
    "tests/video_text/test_v55_geometry.py::test_production_scene_noise_does_not_expand_far_beyond_temporal_anchor",
}


def pytest_collection_modifyitems(config, items):
    if EXTERNAL_VIDEO.exists():
        return
    skip = pytest.mark.skip(
        reason="requires external evaluation video: AI Engineer test.mp4"
    )
    for item in items:
        if item.nodeid in _EXTERNAL_VIDEO_TESTS:
            item.add_marker(skip)
