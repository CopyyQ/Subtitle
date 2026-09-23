import cv2
import numpy as np

from src.video_text.text_enhancement import (
    enhance_white_black_text,
    white_black_text_mask,
)


def test_enhancement_recovers_low_contrast_white_core_with_dark_outline():
    img=np.full((90,180,3),118,np.uint8)
    cv2.rectangle(img,(42,24),(138,66),(55,55,55),-1)
    cv2.rectangle(img,(48,30),(132,60),(168,168,168),-1)

    enhanced=enhance_white_black_text(img)
    mask=white_black_text_mask(img)

    assert enhanced.shape==img.shape[:2]
    assert enhanced.dtype==np.uint8
    # The original core is below the old hard threshold 180, but enhancement
    # must still recover most of it.
    core=mask[32:58,50:130]
    assert core.mean() > .55


def test_enhancement_rejects_flat_bright_patch_without_dark_outline():
    img=np.full((90,180,3),118,np.uint8)
    cv2.rectangle(img,(48,30),(132,60),(220,220,220),-1)

    mask=white_black_text_mask(img)

    assert mask[32:58,50:130].mean() < .10


def test_enhancement_preserves_thin_horizontal_leader():
    img=np.full((90,180,3),110,np.uint8)
    cv2.rectangle(img,(20,41),(70,49),(45,45,45),-1)
    cv2.rectangle(img,(24,43),(66,47),(185,185,185),-1)

    mask=white_black_text_mask(img)

    assert mask[43:48,24:67].mean() > .35


def test_enhancement_does_not_turn_bright_background_gradient_into_text():
    x=np.linspace(120,230,180,dtype=np.uint8)
    gray=np.tile(x,(90,1))
    img=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)

    mask=white_black_text_mask(img)

    assert mask.mean() < .03
