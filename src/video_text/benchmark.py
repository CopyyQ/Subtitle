from __future__ import annotations

import hashlib
import os
import platform
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import torch


def _git_head(path: Path):
    try:
        p=subprocess.run(
            ["git","-C",str(path),"rev-parse","HEAD"],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=3,
        )
        return p.stdout.strip() if p.returncode==0 else None
    except Exception:
        return None


def _sha256(path: Path):
    try:
        h=hashlib.sha256()
        with Path(path).open("rb") as f:
            for chunk in iter(lambda:f.read(1024*1024),b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":",1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def _total_ram_bytes():
    try:
        return int(os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES"))
    except Exception:
        return None


def collect_environment_metadata(project_root: Path, checkpoint: Path) -> dict:
    root=Path(project_root)
    checkpoint=Path(checkpoint)
    try:
        cuda_available=bool(torch.cuda.is_available())
    except Exception:
        cuda_available=False
    try:
        gpu_name=torch.cuda.get_device_name(0) if cuda_available else None
    except Exception:
        gpu_name=None
    fast_root=root/"model"/"FAST"
    return {
        "platform":platform.platform(),
        "os":platform.system(),
        "python_version":sys.version.split()[0],
        "torch_version":str(torch.__version__),
        "opencv_version":str(cv2.__version__),
        "cuda_version":torch.version.cuda,
        "cuda_available":cuda_available,
        "gpu_name":gpu_name,
        "cpu_name":_cpu_name(),
        "cpu_logical_count":os.cpu_count(),
        "ram_total_bytes":_total_ram_bytes(),
        "git_commit":_git_head(root),
        "fast_commit":_git_head(fast_root) if fast_root.exists() else None,
        "checkpoint_name":checkpoint.name,
        "checkpoint_sha256":_sha256(checkpoint) if checkpoint.exists() else None,
    }


def latency_summary(samples_ms: list[float]) -> dict:
    if not samples_ms:
        return {"mean_ms":None,"p50_ms":None,"p95_ms":None}
    arr=np.asarray(samples_ms,dtype=np.float64)
    return {
        "mean_ms":float(np.mean(arr)),
        "p50_ms":float(np.percentile(arr,50)),
        "p95_ms":float(np.percentile(arr,95)),
    }


def real_time_factor(elapsed_seconds: float, duration_seconds: float):
    duration=float(duration_seconds)
    if duration<=0:
        return None
    return float(elapsed_seconds)/duration
