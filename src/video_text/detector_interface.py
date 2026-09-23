from __future__ import annotations
from typing import Protocol
import numpy as np
from .types import Candidate

class DetectorBackend(Protocol):
    def detect_batch(self, images: list[np.ndarray]) -> list[tuple[list[Candidate], list[Candidate]]]:
        ...
