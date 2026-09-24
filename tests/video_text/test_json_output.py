import json

from src.video_text.io import write_coordinate_json


def test_coordinate_json_contains_every_frame_and_ppocr_source(tmp_path):
    path=tmp_path/"coords.json"
    write_coordinate_json(
        path,
        {"fps":30.0,"detector_name":"PP-OCRv5_mobile_det"},
        [
            {
                "frame":1,
                "track_id":7,
                "line_id":0,
                "bbox":[10,20,40,50],
                "source":"PP_HIGH",
                "reconstructed":False,
            }
        ],
        [],
        frame_count=3,
        fps=30.0,
    )
    data=json.loads(path.read_text())
    assert [row["frame"] for row in data["frames"]]==[0,1,2]
    assert data["frames"][0]["boxes"]==[]
    assert data["frames"][1]["boxes"][0]["source"]=="PP_HIGH"
    assert data["frames"][2]["boxes"]==[]
