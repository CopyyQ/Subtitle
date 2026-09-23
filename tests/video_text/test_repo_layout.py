from pathlib import Path


ROOT=Path(__file__).resolve().parents[2]


def test_production_repo_has_no_legacy_four_model_benchmark_layout():
    critical=[
        ROOT/"src/video_text/pipeline.py",
        ROOT/"scripts/setup_fast.sh",
        ROOT/"README.md",
        ROOT/"docs/REPRODUCIBILITY.md",
        ROOT/".gitignore",
    ]
    for path in critical:
        text=path.read_text(encoding="utf-8")
        assert "benchmark_4_scene_models" not in text, path

    assert not (ROOT/"benchmark_4_scene_models").exists()
    assert not (ROOT/"docs/MODEL_SELECTION.md").exists()



def test_v1_reference_artifact_uses_production_output_path():
    import json

    data=json.loads(
        (ROOT/"artifacts/FAST_temporal_v1_consistent_final.json").read_text(
            encoding="utf-8"
        )
    )
    assert data["metadata"]["output"] == "outputs/FAST_temporal_v1_consistent_final.mp4"
