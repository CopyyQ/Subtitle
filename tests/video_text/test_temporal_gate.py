import numpy as np

from src.video_text.temporal_gate import AdaptiveSubtitleGate, TemporalGateConfig


def frame(value=0):
    return np.full((120, 200, 3), value, dtype=np.uint8)


def test_gate_detects_first_frame_then_skips_stable_frames_until_refresh():
    gate = AdaptiveSubtitleGate(
        TemporalGateConfig(max_skip_frames=2, change_threshold=0.02)
    )

    decisions = [gate.should_detect(frame(20), i) for i in range(4)]

    assert decisions == [True, False, False, True]


def test_gate_detects_immediately_when_subtitle_band_changes():
    gate = AdaptiveSubtitleGate(
        TemporalGateConfig(max_skip_frames=5, change_threshold=0.02)
    )
    base = frame(20)
    changed = base.copy()
    changed[40:80, 60:140] = 240

    assert gate.should_detect(base, 0) is True
    assert gate.should_detect(base, 1) is False
    assert gate.should_detect(changed, 2) is True


def test_gate_is_conservative_on_moving_background_outside_caption_band():
    gate = AdaptiveSubtitleGate(
        TemporalGateConfig(
            max_skip_frames=5,
            change_threshold=0.02,
            band_top_fraction=.25,
            band_bottom_fraction=.55,
        )
    )
    base = frame(20)
    outside = base.copy()
    outside[:20, :] = 240
    outside[100:, :] = 240

    assert gate.should_detect(base, 0) is True
    assert gate.should_detect(outside, 1) is False


def test_gate_reset_forces_next_detection():
    gate = AdaptiveSubtitleGate(TemporalGateConfig())
    x = frame(20)
    assert gate.should_detect(x, 0) is True
    assert gate.should_detect(x, 1) is False
    gate.reset()
    assert gate.should_detect(x, 2) is True



def test_gate_detects_small_subtitle_toggle_via_bright_occupancy_delta():
    gate = AdaptiveSubtitleGate(
        TemporalGateConfig(
            max_skip_frames=5,
            change_threshold=.02,
            bright_net_threshold=.0075,
        )
    )
    base = frame(20)
    changed = base.copy()
    changed[45:49, 80:100] = 240

    assert gate.should_detect(base, 0) is True
    assert gate.should_detect(base, 1) is False
    assert gate.should_detect(changed, 2) is True


def test_gate_crop_before_grayscale_matches_reference_features():
    import cv2
    rng=np.random.default_rng(1234)
    roi=rng.integers(0,256,size=(576,720,3),dtype=np.uint8)
    cfg=TemporalGateConfig(
        band_top_fraction=.15,
        band_bottom_fraction=.50,
        signature_width=160,
    )
    gate=AdaptiveSubtitleGate(cfg)
    signature,bright=gate._features(roi)

    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    h,w=gray.shape[:2]
    y1=int(round(h*cfg.band_top_fraction))
    y2=int(round(h*cfg.band_bottom_fraction))
    band=gray[max(0,y1):max(y1+1,y2)]
    target_w=min(cfg.signature_width,max(1,w))
    target_h=max(1,int(round(band.shape[0]*target_w/max(w,1))))
    small=cv2.resize(band,(target_w,target_h),interpolation=cv2.INTER_AREA)
    expected_bright=float(np.mean(small>=150))
    expected_signature=cv2.GaussianBlur(small,(3,3),0).astype(np.int16)

    assert np.array_equal(signature,expected_signature)
    assert bright==expected_bright
