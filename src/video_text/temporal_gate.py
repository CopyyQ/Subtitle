from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class TemporalGateConfig:
    max_skip_frames: int = 2
    change_threshold: float = .02
    bright_net_threshold: float = .0075
    band_top_fraction: float = .25
    band_bottom_fraction: float = .55
    signature_width: int = 160

    def __post_init__(self):
        if self.max_skip_frames < 0:
            raise ValueError("max_skip_frames must be non-negative")
        if not 0.0 <= self.change_threshold <= 1.0:
            raise ValueError("change_threshold must be within 0..1")
        if not 0.0 <= self.bright_net_threshold <= 1.0:
            raise ValueError("bright_net_threshold must be within 0..1")
        if not 0.0 <= self.band_top_fraction < self.band_bottom_fraction <= 1.0:
            raise ValueError("caption band fractions must satisfy 0 <= top < bottom <= 1")
        if self.signature_width <= 0:
            raise ValueError("signature_width must be positive")


class AdaptiveSubtitleGate:
    """Conservative frame gate for stable hard-coded subtitle regions.

    It compares a compact grayscale signature from the caption band against
    the last frame that was actually sent to the detector. A periodic refresh
    bounds the maximum number of reused frames even when the visual change
    score stays below threshold.
    """

    def __init__(self, config: TemporalGateConfig | None = None):
        self.config = config or TemporalGateConfig()
        self._reference = None
        self._reference_bright_fraction = None
        self._skipped_since_detect = 0

    def reset(self):
        self._reference = None
        self._reference_bright_fraction = None
        self._skipped_since_detect = 0

    def _features(self, roi: np.ndarray) -> tuple[np.ndarray, float]:
        if roi.ndim == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = np.asarray(roi, dtype=np.uint8)
        h, w = gray.shape[:2]
        y1 = int(round(h * self.config.band_top_fraction))
        y2 = int(round(h * self.config.band_bottom_fraction))
        band = gray[max(0, y1):max(y1 + 1, y2)]
        target_w = min(self.config.signature_width, max(1, w))
        target_h = max(1, int(round(band.shape[0] * target_w / max(w, 1))))
        small = cv2.resize(
            band,
            (target_w, target_h),
            interpolation=cv2.INTER_AREA,
        )
        bright_fraction = float(np.mean(small >= 150))
        # Light blur suppresses codec noise while preserving subtitle changes.
        signature = cv2.GaussianBlur(small, (3, 3), 0).astype(np.int16)
        return signature, bright_fraction

    def _signature(self, roi: np.ndarray) -> np.ndarray:
        return self._features(roi)[0]

    def should_detect(self, roi: np.ndarray, frame_index: int) -> bool:
        signature, bright_fraction = self._features(roi)
        if self._reference is None:
            self._reference = signature
            self._reference_bright_fraction = bright_fraction
            self._skipped_since_detect = 0
            return True

        change = float(np.mean(np.abs(signature - self._reference))) / 255.0
        bright_net = abs(
            bright_fraction - float(self._reference_bright_fraction)
        )
        bright_transition = (
            self.config.bright_net_threshold > 0.0
            and bright_net >= self.config.bright_net_threshold
        )
        periodic_refresh = (
            self._skipped_since_detect >= self.config.max_skip_frames
        )
        if (
            change >= self.config.change_threshold
            or bright_transition
            or periodic_refresh
        ):
            self._reference = signature
            self._reference_bright_fraction = bright_fraction
            self._skipped_since_detect = 0
            return True

        self._skipped_since_detect += 1
        return False
