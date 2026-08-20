import numpy as np
from edge.railguard_edge.features import vibration_features, vision_features

def test_sine_rms_and_bands():
    fs=2000; t=np.arange(fs)/fs; x=np.sin(2*np.pi*100*t)
    f=vibration_features(x,fs)
    assert 0.69 < f["rms_ms2"] < 0.72
    assert len(f["band_energy"])==4
    assert abs(sum(f["band_energy"])-1.0)<1e-6


def test_sharpness_metric_distinguishes_edges_from_blur():
    import cv2
    sharp=np.zeros((96,96),dtype=np.uint8)
    sharp[:,::8]=255
    blurred=cv2.GaussianBlur(sharp,(11,11),3.0)
    a=vision_features(sharp); b=vision_features(blurred)
    assert a["sharpness"] > b["sharpness"]
    assert a["sharpness"] >= 0 and b["sharpness"] >= 0
