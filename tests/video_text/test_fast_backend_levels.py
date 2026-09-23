import numpy as np
import torch

from src.video_text.fast_backend import FastBackend, ScoredComponent, split_levels


def c(score, area):
    return ScoredComponent(np.array([10,10,100,40], np.float32), score, area)


def test_high_candidate_is_not_duplicated_as_low():
    high, low = split_levels([c(.93, 300)], .88, .75, 250, 140)
    assert len(high) == 1
    assert len(low) == 0
    assert high[0].level == "HIGH"


def test_weak_supported_component_goes_to_low():
    high, low = split_levels([c(.79, 180)], .88, .75, 250, 140)
    assert len(high) == 0
    assert len(low) == 1
    assert low[0].level == "LOW"


def test_component_below_low_threshold_is_discarded():
    high, low = split_levels([c(.70, 500)], .88, .75, 250, 140)
    assert high == []
    assert low == []


class FakeBackend(FastBackend):
    def __init__(self):
        self.high_score=.88
        self.low_score=.75
        self.high_min_area=250
        self.low_min_area=140
        self.forward_calls=0

    def _prep(self, images):
        return torch.zeros((len(images), 3, 8, 8)), [(8,8)]*len(images), [(8,8)]*len(images)

    def _forward_model(self, x):
        self.forward_calls += 1
        return object()

    def _decode_scored_components(self, out, origs, sizes):
        return [[c(.93,300), c(.79,180)] for _ in origs]


def test_detect_batch_uses_one_forward_for_high_and_low():
    backend=FakeBackend()
    results=backend.detect_batch([
        np.zeros((8,8,3),np.uint8),
        np.zeros((8,8,3),np.uint8),
    ])
    assert backend.forward_calls == 1
    assert len(results) == 2
    assert len(results[0][0]) == 1
    assert len(results[0][1]) == 1
