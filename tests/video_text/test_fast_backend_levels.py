import inspect
from types import SimpleNamespace

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


def test_fast_backend_constructor_exposes_runtime_options():
    params=inspect.signature(FastBackend.__init__).parameters
    assert "precision" in params
    assert "batch_size" in params
    assert "pin_memory" in params


def test_forward_uses_fp16_autocast_only_for_cuda(monkeypatch):
    calls=[]
    class Chain:
        def backbone(self,x):
            return x
        def neck(self,x):
            return x
        def det_head(self,x):
            return x
    backend=FastBackend.__new__(FastBackend)
    backend.device=torch.device("cuda")
    backend.precision="fp16"
    backend.forward_calls=0
    backend.model=Chain()

    class Context:
        def __enter__(self):
            return None
        def __exit__(self,*args):
            return False

    def fake_autocast(**kwargs):
        calls.append(kwargs)
        return Context()

    monkeypatch.setattr(torch,"autocast",fake_autocast)
    out=backend._forward_model(torch.ones((1,1,2,2)))
    assert torch.equal(out,torch.ones((1,1,2,2)))
    assert calls==[{"device_type":"cuda","dtype":torch.float16,"enabled":True}]


def test_fp32_fp16_decode_has_same_candidate_counts_for_fixed_logits():
    backend=FastBackend.__new__(FastBackend)
    backend.model=SimpleNamespace(det_head=SimpleNamespace(pooling_size=2))
    backend.low_min_area=1
    backend.low_score=0.0
    backend.high_min_area=1
    backend.high_score=0.5
    logits=torch.full((1,1,4,4),-4.0)
    logits[:,:,1:3,1:3]=4.0
    args=([(16,16)],[(16,16)])
    comps32=backend._decode_scored_components(logits.float(),*args)
    comps16=backend._decode_scored_components(logits.half(),*args)
    levels32=split_levels(comps32[0],.5,0.0,1,1)
    levels16=split_levels(comps16[0],.5,0.0,1,1)
    assert tuple(map(len,levels32))==tuple(map(len,levels16))
