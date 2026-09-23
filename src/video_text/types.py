from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np


@dataclass(slots=True)
class Candidate:
    bbox: np.ndarray
    score: float
    level: str
    source: str = "fast"

    def __post_init__(self):
        self.bbox = np.asarray(self.bbox, dtype=np.float32)
        self.score = float(self.score)
        if self.level not in {"HIGH", "LOW"}:
            raise ValueError(f"invalid candidate level: {self.level}")


@dataclass(slots=True)
class FrameDetections:
    frame_index: int
    timestamp: float
    high: list[Candidate] = field(default_factory=list)
    low: list[Candidate] = field(default_factory=list)


@dataclass(slots=True)
class TrackObservation:
    frame_index: int
    bbox: np.ndarray
    score: float | None
    level: str
    reconstructed: bool = False

    def __post_init__(self):
        self.bbox = np.asarray(self.bbox, dtype=np.float32)


@dataclass(slots=True)
class SubtitleTrack:
    track_id: int
    observations: dict[int, TrackObservation] = field(default_factory=dict)
    confirmed: bool = False
    language_status: str = "unchecked"
    content_boundary_before: bool = False

    def sorted_frames(self) -> list[int]:
        return sorted(self.observations)
