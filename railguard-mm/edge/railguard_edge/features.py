from __future__ import annotations
import numpy as np
from scipy.stats import kurtosis


def vibration_features(samples: np.ndarray, sample_rate_hz: float) -> dict:
    """Compute compact features from N x C acceleration samples in m/s^2."""
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim == 2:
        mag = np.linalg.norm(x, axis=1)
    else:
        mag = x.reshape(-1)
    mag = mag - np.mean(mag)
    rms = float(np.sqrt(np.mean(mag ** 2) + 1e-12))
    peak = float(np.max(np.abs(mag)))
    k = float(kurtosis(mag, fisher=False, bias=False)) if len(mag) > 8 else 3.0
    crest = float(peak / (rms + 1e-12))
    freqs = np.fft.rfftfreq(len(mag), d=1.0 / sample_rate_hz)
    power = np.abs(np.fft.rfft(mag)) ** 2
    denom = float(np.sum(power) + 1e-12)
    centroid = float(np.sum(freqs * power) / denom)
    nyq = sample_rate_hz / 2.0
    edges = np.linspace(0, nyq, 5)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        bands.append(float(np.sum(power[mask]) / denom))
    return {
        "rms_ms2": rms,
        "peak_ms2": peak,
        "kurtosis": k,
        "crest_factor": crest,
        "spectral_centroid_hz": centroid,
        "band_energy": bands,
    }


def vision_features(gray: np.ndarray, prev_gray: np.ndarray | None = None) -> dict:
    import cv2
    g = np.asarray(gray)
    if g.ndim == 3:
        g = cv2.cvtColor(g, cv2.COLOR_BGR2GRAY)
    g = g.astype(np.uint8)
    contrast = float(np.std(g) / 64.0)
    sharpness = float(cv2.Laplacian(g, cv2.CV_64F).var() / 1000.0)
    motion = 0.0
    if prev_gray is not None:
        p = prev_gray
        if p.ndim == 3:
            p = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(p, g, None, 0.5, 3, 15, 3, 5, 1.1, 0)
        motion = float(np.mean(np.linalg.norm(flow, axis=2)))
    return {"motion_score": motion, "contrast": contrast, "sharpness": sharpness}
