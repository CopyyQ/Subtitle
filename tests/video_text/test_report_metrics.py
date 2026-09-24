from pathlib import Path

from src.video_text.benchmark import collect_environment_metadata


def test_environment_metadata_records_detector_and_optional_paddle_versions():
    root=Path(__file__).resolve().parents[2]
    data=collect_environment_metadata(
        root,
        root/"missing-checkpoint.pth",
        detector_name="PP-OCRv5_mobile_det",
    )

    assert data["detector_name"]=="PP-OCRv5_mobile_det"
    assert "paddle_version" in data
    assert "paddleocr_version" in data
