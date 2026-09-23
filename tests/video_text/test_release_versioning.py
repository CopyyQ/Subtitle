from pathlib import Path
import subprocess


ROOT=Path(__file__).resolve().parents[2]


def test_release_tree_uses_v1_namespace_only():
    tracked=subprocess.check_output(
        ["git","ls-files"],cwd=ROOT,text=True
    ).splitlines()
    legacy=[
        "v5"+"_"+"6",
        "v5"+"."+"6",
        "V5"+"."+"6",
        "V5"+"_"+"6",
        "v"+"56",
        "V"+"56",
    ]
    offenders=[]
    for rel in tracked:
        path=ROOT/rel
        if not path.is_file():
            continue
        try:
            text=path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(token in text for token in legacy):
            offenders.append(rel)
        if any(token in path.name for token in legacy):
            offenders.append(rel)
    assert not sorted(set(offenders)), sorted(set(offenders))

    from src.video_text.pipeline import PipelineConfig
    assert PipelineConfig().temporal_mode == "v1"

    assert (ROOT/"src/video_text/v1_geometry.py").is_file()
    assert (ROOT/"src/video_text/v1_processor.py").is_file()
    assert (ROOT/"artifacts/FAST_temporal_v1_consistent_final.json").is_file()
