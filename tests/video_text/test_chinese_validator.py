import numpy as np
from src.video_text.types import SubtitleTrack, TrackObservation
from src.video_text.chinese_validator import cjk_ratio, validate_track

class FakeRecognizer:
    def __init__(self,results):
        self.results=list(results); self.i=0
    def recognize(self,crop):
        r=self.results[min(self.i,len(self.results)-1)]
        self.i+=1
        return r

def make_track():
    t=SubtitleTrack(1,confirmed=True)
    for i in range(3):
        t.observations[i]=TrackObservation(i,np.array([10,10,100,40],np.float32),.95,"HIGH")
    return t

def frames():
    return {i:np.zeros((80,140,3),np.uint8) for i in range(3)}

def test_cjk_ratio_detects_chinese_text():
    assert cjk_ratio("我们开始吧")==1.0

def test_latin_only_has_zero_cjk_ratio():
    assert cjk_ratio("OPENAI")==0.0

def test_clear_non_chinese_track_is_rejected():
    r=FakeRecognizer([("OPENAI",.99),("HELLO",.98),("WORLD",.97)])
    d=validate_track(make_track(),frames(),r,max_samples=3)
    assert d.status=="rejected_non_chinese"

def test_uncertain_ocr_keeps_strong_track():
    r=FakeRecognizer([("",0.0),("",0.0),("",0.0)])
    d=validate_track(make_track(),frames(),r,max_samples=3)
    assert d.status=="retained_uncertain"

def test_chinese_track_is_validated():
    r=FakeRecognizer([("我们开始吧",.95),("我们开始吧",.96),("我们开始吧",.94)])
    d=validate_track(make_track(),frames(),r,max_samples=3)
    assert d.status=="validated_chinese"


def test_weak_track_keeps_repeated_cjk_even_at_low_confidence():
    from src.video_text.chinese_validator import validate_weak_track
    r=FakeRecognizer([("哼",.14),("哼",.11),("哼",.09)])
    d=validate_weak_track(make_track(),frames(),r,max_samples=3)
    assert d.status=="validated_chinese"


def test_weak_track_rejects_persistent_unreadable_region():
    from src.video_text.chinese_validator import validate_weak_track
    r=FakeRecognizer([("",0.0),("",0.0),("",0.0)])
    d=validate_weak_track(make_track(),frames(),r,max_samples=3)
    assert d.status=="rejected_non_chinese"


def test_weak_track_rejects_repeated_latin_only_region():
    from src.video_text.chinese_validator import validate_weak_track
    r=FakeRecognizer([("OPEN",.25),("OPEN",.20),("OPEN",.18)])
    d=validate_weak_track(make_track(),frames(),r,max_samples=3)
    assert d.status=="rejected_non_chinese"


def test_weak_track_rejects_near_zero_confidence_cjk_hallucination():
    from src.video_text.chinese_validator import validate_weak_track
    r=FakeRecognizer([("凡外",.006),("泄厌",.004),("人",.009)])
    d=validate_weak_track(make_track(),frames(),r,max_samples=3)
    assert d.status=="rejected_non_chinese"


def test_weak_track_keeps_small_real_glyph_at_measured_confidence():
    from src.video_text.chinese_validator import validate_weak_track
    r=FakeRecognizer([("哼",.077),("哼",.049),("哼",.048)])
    d=validate_weak_track(make_track(),frames(),r,max_samples=3)
    assert d.status=="validated_chinese"
