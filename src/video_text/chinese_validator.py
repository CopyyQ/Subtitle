from __future__ import annotations
from dataclasses import dataclass
import numpy as np

def _is_cjk(ch):
    cp=ord(ch)
    return (0x3400<=cp<=0x4DBF or 0x4E00<=cp<=0x9FFF or
            0xF900<=cp<=0xFAFF or 0x20000<=cp<=0x2FA1F)

def contains_cjk(text):
    return any(_is_cjk(ch) for ch in text)

def cjk_ratio(text):
    chars=[ch for ch in text if not ch.isspace()]
    if not chars:
        return 0.0
    return sum(_is_cjk(ch) for ch in chars)/len(chars)

@dataclass(frozen=True,slots=True)
class ValidationDecision:
    status: str
    texts: list[str]
    cjk_ratio: float

def representative_frames(track,max_samples=3):
    eligible=[
        f for f in track.sorted_frames()
        if track.observations[f].level=="HIGH" and not track.observations[f].reconstructed
    ]
    if not eligible:
        eligible=track.sorted_frames()
    if not eligible:
        return []
    if len(eligible)<=max_samples:
        return eligible
    ids=np.linspace(0,len(eligible)-1,max_samples).round().astype(int)
    return [eligible[i] for i in sorted(set(ids.tolist()))]

def validate_track(track,frames,recognizer,max_samples=3,min_confidence=.70):
    texts=[]; confident=[]
    for fi in representative_frames(track,max_samples):
        frame=frames.get(fi)
        if frame is None:
            continue
        b=track.observations[fi].bbox
        h,w=frame.shape[:2]
        x1=max(0,int(np.floor(b[0]))); y1=max(0,int(np.floor(b[1])))
        x2=min(w,int(np.ceil(b[2]))); y2=min(h,int(np.ceil(b[3])))
        if x2<=x1 or y2<=y1:
            continue
        text,conf=recognizer.recognize(frame[y1:y2,x1:x2])
        text=str(text or "")
        texts.append(text)
        if conf is not None and float(conf)>=min_confidence and text.strip():
            confident.append(text)
    joined="".join(confident)
    ratio=cjk_ratio(joined)
    if joined and contains_cjk(joined):
        status="validated_chinese"
    elif len(confident)>=2 and all(not contains_cjk(x) for x in confident):
        status="rejected_non_chinese"
    else:
        status="retained_uncertain"
    track.language_status=status
    return ValidationDecision(status,texts,ratio)

class EasyOCRChineseRecognizer:
    def __init__(self,gpu=True):
        import easyocr
        self.reader=easyocr.Reader(["ch_sim","en"],gpu=gpu,verbose=False)

    def recognize(self,crop):
        rows=self.reader.readtext(crop,detail=1,paragraph=False)
        if not rows:
            return "",0.0
        text="".join(str(r[1]) for r in rows)
        conf=float(np.mean([float(r[2]) for r in rows]))
        return text,conf


def validate_weak_track(
    track,frames,recognizer,max_samples=3,min_confidence=.02
):
    texts=[]
    cjk_hits=0
    for fi in representative_frames(track,max_samples):
        frame=frames.get(fi)
        if frame is None:
            continue
        b=track.observations[fi].bbox
        h,w=frame.shape[:2]
        x1=max(0,int(np.floor(b[0]))); y1=max(0,int(np.floor(b[1])))
        x2=min(w,int(np.ceil(b[2]))); y2=min(h,int(np.ceil(b[3])))
        if x2<=x1 or y2<=y1:
            continue
        text,conf=recognizer.recognize(frame[y1:y2,x1:x2])
        text=str(text or "")
        texts.append(text)
        if (
            conf is not None
            and float(conf)>=float(min_confidence)
            and contains_cjk(text)
        ):
            cjk_hits+=1

    joined="".join(texts)
    ratio=cjk_ratio(joined)
    required=2 if len(texts)>=2 else 1
    if cjk_hits>=required:
        status="validated_chinese"
    else:
        status="rejected_non_chinese"
    track.language_status=status
    return ValidationDecision(status,texts,ratio)
