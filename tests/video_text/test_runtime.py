import torch
import pytest

from src.video_text.runtime import (
    configure_runtime,
    resolve_device,
    resolve_precision,
    synchronize,
)


def test_auto_uses_cpu_when_cuda_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto").type == "cpu"


def test_explicit_cpu_wins_even_when_cuda_exists(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cpu").type == "cpu"


def test_explicit_cuda_requires_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        resolve_device("cuda")


def test_fp16_is_cuda_only():
    with pytest.raises(ValueError, match="fp16"):
        resolve_precision("fp16", torch.device("cpu"))


def test_cpu_synchronize_never_calls_cuda_even_when_cuda_exists(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    calls=[]
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: calls.append(args))
    synchronize(torch.device("cpu"))
    assert calls==[]


def test_cpu_runtime_uses_explicit_thread_count(monkeypatch):
    calls=[]
    monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))
    result=configure_runtime(torch.device("cpu"), cpu_threads=6)
    assert result=={"cpu_threads":6}
    assert calls==[6]
