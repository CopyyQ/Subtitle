import numpy as np
import pytest

from src.video_text.ppocrv5_backend import PPOCRv5MobileBackend


class FakePredictor:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def predict(self, images):
        self.calls.append(len(images))
        return self.rows[:len(images)]


def result(polys, scores):
    return {
        "dt_polys": [np.asarray(p, dtype=np.float32) for p in polys],
        "dt_scores": np.asarray(scores, dtype=np.float32),
    }


def test_maps_polygons_to_high_and_low_candidates():
    predictor = FakePredictor([
        result(
            [
                [[10, 10], [110, 10], [110, 40], [10, 40]],
                [[20, 50], [50, 50], [50, 80], [20, 80]],
            ],
            [.93, .61],
        )
    ])
    backend = PPOCRv5MobileBackend(
        device="cpu",
        batch_size=4,
        high_score=.84,
        low_score=.50,
        predictor=predictor,
    )

    high, low = backend.detect_batch(
        [np.zeros((100, 200, 3), np.uint8)]
    )[0]

    assert high[0].bbox.tolist() == [10, 10, 110, 40]
    assert low[0].bbox.tolist() == [20, 50, 50, 80]
    assert high[0].source == "ppocrv5_mobile"
    assert low[0].source == "ppocrv5_mobile"


def test_drops_scores_below_low_threshold_and_preserves_batch_length():
    predictor = FakePredictor([
        result([[[1, 2], [8, 2], [8, 9], [1, 9]]], [.49]),
        result([[[2, 3], [9, 3], [9, 10], [2, 10]]], [.99]),
    ])
    backend = PPOCRv5MobileBackend(
        device="cpu",
        batch_size=2,
        high_score=.84,
        low_score=.50,
        predictor=predictor,
    )

    rows = backend.detect_batch([
        np.zeros((20, 20, 3), np.uint8),
        np.zeros((20, 20, 3), np.uint8),
    ])

    assert len(rows) == 2
    assert rows[0] == ([], [])
    assert len(rows[1][0]) == 1
    assert rows[1][1] == []


def test_rejects_predictor_batch_length_mismatch():
    backend = PPOCRv5MobileBackend(
        device="cpu",
        predictor=FakePredictor([]),
    )
    with pytest.raises(RuntimeError, match="batch length"):
        backend.detect_batch([np.zeros((20, 20, 3), np.uint8)])


def test_threshold_validation():
    with pytest.raises(ValueError, match="low_score"):
        PPOCRv5MobileBackend(
            device="cpu", high_score=.4, low_score=.5,
            predictor=FakePredictor([]),
        )
