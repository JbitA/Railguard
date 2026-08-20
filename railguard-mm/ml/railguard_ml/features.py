from __future__ import annotations
import numpy as np
from scipy.stats import kurtosis, skew


def window_features(x: np.ndarray, fs: float) -> dict[str, float]:
    x = np.asarray(x, dtype=float).reshape(-1)
    x = x - np.mean(x)
    rms = np.sqrt(np.mean(x*x) + 1e-12)
    peak = np.max(np.abs(x))
    spec = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1/fs)
    s = np.sum(spec) + 1e-12
    return {
        "rms": float(rms),
        "std": float(np.std(x)),
        "peak": float(peak),
        "ptp": float(np.ptp(x)),
        "kurtosis": float(kurtosis(x, fisher=False, bias=False)),
        "skew": float(skew(x, bias=False)),
        "crest": float(peak/(rms+1e-12)),
        "spectral_centroid": float(np.sum(f*spec)/s),
    }
