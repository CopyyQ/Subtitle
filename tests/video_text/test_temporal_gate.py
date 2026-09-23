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
