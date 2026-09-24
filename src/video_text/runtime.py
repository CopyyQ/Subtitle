from __future__ import annotations

import os

import torch


def resolve_device(requested: str) -> torch.device:
    requested = str(requested).lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError("device must be auto, cuda, or cpu")


def resolve_precision(requested: str, device: torch.device) -> str:
    requested = str(requested).lower()
    if requested == "auto":
        return "fp16" if device.type == "cuda" else "fp32"
    if requested not in {"fp32", "fp16"}:
        raise ValueError("precision must be auto, fp32, or fp16")
    if requested == "fp16" and device.type != "cuda":
        raise ValueError("fp16 is supported only on CUDA")
    return requested


def configure_runtime(device: torch.device, cpu_threads: int | None = None) -> dict:
    if device.type == "cpu":
        threads = int(cpu_threads or max(1, (os.cpu_count() or 1) // 2))
        torch.set_num_threads(threads)
        return {"cpu_threads": threads}
    torch.backends.cudnn.benchmark = True
    return {"cpu_threads": 0}


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
