from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


@dataclass(frozen=True)
class PairedBootstrapResult:
    groups: int
    estimate: float
    ci95_low: float | None
    ci95_high: float | None
    probability_improvement: float | None
    repeats: int

    def to_dict(self) -> dict:
        return {
            "groups": self.groups,
            "estimate": self.estimate,
            "ci95_low": self.ci95_low,
            "ci95_high": self.ci95_high,
            "probability_improvement": self.probability_improvement,
            "repeats": self.repeats,
        }


def paired_group_bootstrap(
    baseline: Mapping[str, float],
    candidate: Mapping[str, float],
    *,
    repeats: int = 5000,
    seed: int = 17,
) -> PairedBootstrapResult:
    """Bootstrap an equal-weight, paired held-out-group error improvement.

    The unit of resampling is a geographic block/run rather than an individual
    temporal window.  That avoids presenting a naive IID-window confidence interval
    for strongly autocorrelated time-series data.

    Positive values mean the candidate has lower error than the baseline.
    """
    if repeats <= 0:
        raise ValueError("bootstrap repeats must be positive")
    groups = sorted(set(baseline) & set(candidate))
    if not groups:
        raise ValueError("paired bootstrap requires at least one common held-out group")
    b = np.asarray([float(baseline[g]) for g in groups], dtype=np.float64)
    c = np.asarray([float(candidate[g]) for g in groups], dtype=np.float64)
    if not np.isfinite(b).all() or not np.isfinite(c).all():
        raise ValueError("paired bootstrap inputs must be finite")
    delta = b - c
    estimate = float(delta.mean())

    # With fewer than three independent geographic units, percentile intervals are
    # so unstable that publishing them adds false precision. Preserve the point
    # estimate but mark uncertainty as unavailable.
    if len(groups) < 3:
        return PairedBootstrapResult(len(groups), estimate, None, None, None, repeats)

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(repeats, len(groups)))
    samples = delta[draws].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return PairedBootstrapResult(
        groups=len(groups),
        estimate=estimate,
        ci95_low=float(low),
        ci95_high=float(high),
        probability_improvement=float(np.mean(samples > 0.0)),
        repeats=int(repeats),
    )
