from __future__ import annotations

import cv2
import numpy as np


def _gray(image):
    if image.ndim==2:
        return image.astype(np.uint8,copy=False)
    return cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)


def enhance_white_black_text(
    image,
    clahe_clip=2.5,
    clahe_grid=(8,8),
    unsharp_sigma=1.0,
    unsharp_amount=.65,
):
    """Enhance white subtitle cores while preserving their dark outline.

    This is intended for already-cropped subtitle lanes, not a whole scene.
    CLAHE normalizes local illumination; a restrained unsharp step restores
    thin white strokes without inventing large bright regions.
    """
    gray=_gray(image)
    clahe=cv2.createCLAHE(
        clipLimit=float(clahe_clip),
        tileGridSize=tuple(int(x) for x in clahe_grid),
    )
    local=clahe.apply(gray)
    blur=cv2.GaussianBlur(local,(0,0),float(unsharp_sigma))
    sharp=cv2.addWeighted(
        local,
        1.0+float(unsharp_amount),
        blur,
        -float(unsharp_amount),
        0,
    )
    return np.clip(sharp,0,255).astype(np.uint8)


def white_black_text_mask(
    image,
    bright_threshold=150,
    min_original_bright=125,
    dark_threshold=100,
    outline_radius=3,
    local_contrast_threshold=22,
):
    """Return a binary mask for white-core / dark-outline subtitle pixels.

    A bright connected component is accepted only when at least part of it is
    adjacent to genuinely dark pixels. This lets enhancement recover dim white
    strokes while rejecting flat bright clothing/background regions.
    """
    gray=_gray(image)
    enhanced=enhance_white_black_text(image)

    k=max(3,2*int(outline_radius)+1)
    local_min=cv2.erode(
        enhanced,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)),
    )
    contrast=cv2.subtract(enhanced,local_min)

    candidate=(
        ((enhanced>=int(bright_threshold)) & (gray>=int(min_original_bright)))
        | ((contrast>=int(local_contrast_threshold)) & (gray>=int(min_original_bright)))
    ).astype(np.uint8)
    candidate=cv2.morphologyEx(
        candidate,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT,(3,3)),
    )

    dark=(gray<=int(dark_threshold)).astype(np.uint8)
    near_dark=cv2.dilate(
        dark,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)),
    )
    seed=(candidate & near_dark).astype(np.uint8)

    n,labels,stats,_=cv2.connectedComponentsWithStats(candidate,8)
    out=np.zeros_like(candidate)
    for i in range(1,n):
        area=int(stats[i,cv2.CC_STAT_AREA])
        if area<2:
            continue
        component=(labels==i)
        seed_count=int(seed[component].sum())
        if seed_count<2:
            continue
        # Require at least a small fraction of the bright component to touch
        # the dark outline. Real white glyph strokes satisfy this naturally.
        if seed_count/max(area,1) < .015:
            continue
        out[component]=1

    return out.astype(np.uint8)
