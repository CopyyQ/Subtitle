from __future__ import annotations

from dataclasses import dataclass
import cv2
import numpy as np

from .text_enhancement import white_black_text_mask


@dataclass(slots=True)
class SlotPrior:
    y1: int
    y2: int
    expected_x_center: float
    expected_height: float
    expected_x1: float | None = None
    expected_x2: float | None = None


@dataclass(slots=True)
class WeakBirthRecord:
    frame: int
    bbox: np.ndarray
    confirmed: bool = True


def _iou(a, b):
    x1=max(float(a[0]),float(b[0]))
    y1=max(float(a[1]),float(b[1]))
    x2=min(float(a[2]),float(b[2]))
    y2=min(float(a[3]),float(b[3]))
    inter=max(0.0,x2-x1)*max(0.0,y2-y1)
    aa=max(0.0,float(a[2]-a[0]))*max(0.0,float(a[3]-a[1]))
    bb=max(0.0,float(b[2]-b[0]))*max(0.0,float(b[3]-b[1]))
    den=aa+bb-inter
    return inter/den if den>0 else 0.0


def detect_weak_text_in_slot(
    frame,
    prior: SlotPrior,
    white_threshold: int = 180,
    dark_threshold: int = 90,
    outline_radius: int = 3,
    center_tolerance_heights: float = 1.0,
    outline_pad_ratio: float = .12,
):
    """Find a small subtitle glyph in an already learned vertical slot.

    This is deliberately only a weak-birth detector. It uses the video's
    white-core + dark-outline appearance and a learned subtitle center/font
    height. It is not a replacement for FAST on normal text lines.
    """
    fh,fw=frame.shape[:2]
    h=max(1.0,float(prior.expected_height))
    y1=max(0,int(prior.y1))
    y2=min(fh,int(prior.y2))
    half=max(int(round(2.0*h)),32)
    cx=float(prior.expected_x_center)
    x1=max(0,int(np.floor(cx-half)))
    x2=min(fw,int(np.ceil(cx+half)))
    if x2<=x1 or y2<=y1:
        return None

    crop=frame[y1:y2,x1:x2]
    core=white_black_text_mask(
        crop,
        bright_threshold=min(160,int(white_threshold)),
        dark_threshold=max(95,int(dark_threshold)),
        outline_radius=int(outline_radius),
    )

    n,labels,stats,centroids=cv2.connectedComponentsWithStats(core,8)
    parts=[]
    for i in range(1,n):
        px,py,pw,ph,area=stats[i]
        if area < max(8,int(round(h*h*.004))):
            continue
        if pw<2 or ph<3:
            continue
        if ph > 1.35*h or pw > 1.60*h:
            continue
        pcx=x1+float(centroids[i][0])
        pcy=y1+float(centroids[i][1])
        if abs(pcx-cx) > float(center_tolerance_heights)*h:
            continue
        parts.append([
            float(x1+px),float(y1+py),
            float(x1+px+pw),float(y1+py+ph),
            float(area),
        ])
    if not parts:
        return None

    # A single Chinese glyph may be split into several white-core components.
    # Keep only parts close to the learned center, then union them.
    parts.sort(key=lambda p: abs(((p[0]+p[2])*.5)-cx))
    seed=parts[0]
    selected=[seed]
    union=np.array(seed[:4],np.float32)
    for part in parts[1:]:
        box=np.array(part[:4],np.float32)
        gap_x=max(0.0,max(float(union[0]),float(box[0]))-min(float(union[2]),float(box[2])))
        gap_y=max(0.0,max(float(union[1]),float(box[1]))-min(float(union[3]),float(box[3])))
        if gap_x <= .35*h and gap_y <= .35*h:
            selected.append(part)
            union=np.array([
                min(float(union[0]),float(box[0])),
                min(float(union[1]),float(box[1])),
                max(float(union[2]),float(box[2])),
                max(float(union[3]),float(box[3])),
            ],np.float32)

    uh=float(union[3]-union[1])
    uw=float(union[2]-union[0])
    ucx=float((union[0]+union[2])*.5)
    if uh < .45*h or uh > 1.35*h:
        return None
    if uw < .18*h or uw > 1.60*h:
        return None
    if abs(ucx-cx) > .80*h:
        return None
    if sum(p[4] for p in selected) < max(25.0,h*h*.02):
        return None

    pad=max(2,int(round(h*float(outline_pad_ratio))))
    union[0]=max(0.0,float(union[0])-pad)
    union[1]=max(0.0,float(union[1])-pad)
    union[2]=min(float(fw),float(union[2])+pad)
    union[3]=min(float(fh),float(union[3])+pad)
    return union.astype(np.float32)


class WeakBirthTracker:
    def __init__(self,confirm_frames=3,max_gap=0,iou_gate=.45):
        self.confirm_frames=int(confirm_frames)
        self.max_gap=int(max_gap)
        self.iou_gate=float(iou_gate)
        self._pending=[]
        self._last_frame=None
        self._canonical=None
        self._active=False

    def _reset_pending(self):
        self._pending=[]
        self._last_frame=None

    def update(self,frame,box):
        frame=int(frame)
        if box is None:
            if (
                self._last_frame is None
                or frame-self._last_frame > self.max_gap
            ):
                self._reset_pending()
                self._active=False
                self._canonical=None
            return []

        box=np.asarray(box,dtype=np.float32).copy()

        if self._active:
            if _iou(self._canonical,box) >= self.iou_gate:
                self._last_frame=frame
                return [WeakBirthRecord(frame,self._canonical.copy(),True)]
            self._active=False
            self._canonical=None
            self._reset_pending()

        if self._pending:
            prev_frame,prev_box=self._pending[-1]
            compatible=(
                frame-prev_frame <= self.max_gap+1
                and _iou(prev_box,box) >= self.iou_gate
            )
            if not compatible:
                self._reset_pending()

        self._pending.append((frame,box))
        self._last_frame=frame

        if len(self._pending) < self.confirm_frames:
            return []

        boxes=np.stack([b for _,b in self._pending],axis=0)
        canonical=np.median(boxes,axis=0).astype(np.float32)
        self._canonical=canonical
        self._active=True
        rows=[
            WeakBirthRecord(f,canonical.copy(),True)
            for f,_ in self._pending
        ]
        self._pending=[]
        return rows

    def flush(self):
        self._reset_pending()
        return []
