from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    fourcc: str


def bottom_roi(height: int, fraction: float) -> tuple[int, int]:
    if not 0.25 <= float(fraction) <= 0.45:
        raise ValueError("roi_bottom_fraction must be within 0.25..0.45")
    y1 = int(round(height * (1.0 - float(fraction))))
    return max(0, min(height - 1, y1)), height


def probe_video(path: str | Path) -> VideoInfo:
    p = Path(path)
    if p.suffix.lower() != ".mp4":
        raise ValueError("input must be an .mp4 file")
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise RuntimeError(f"cannot decode MP4: {p}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    code = int(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()
    if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
        raise RuntimeError(f"invalid MP4 stream metadata: {p}")
    fourcc = "".join(chr((code >> (8 * i)) & 0xFF) for i in range(4))
    return VideoInfo(width, height, fps, frame_count, fourcc)
