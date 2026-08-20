from __future__ import annotations

import pytest

from ml.railguard_ml.statistics import paired_group_bootstrap


def test_paired_group_bootstrap_reports_positive_candidate_improvement():
    baseline = {"a": 2.0, "b": 3.0, "c": 4.0, "d": 5.0}
    candidate = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}
    result = paired_group_bootstrap(baseline, candidate, repeats=1000, seed=1)
    assert result.groups == 4
    assert result.estimate == pytest.approx(1.0)
    assert result.ci95_low == pytest.approx(1.0)
    assert result.ci95_high == pytest.approx(1.0)
    assert result.probability_improvement == 1.0


def test_paired_group_bootstrap_refuses_false_precision_for_tiny_group_count():
    result = paired_group_bootstrap({"a": 2.0, "b": 2.0}, {"a": 1.0, "b": 3.0})
    assert result.groups == 2
    assert result.estimate == pytest.approx(0.0)
    assert result.ci95_low is None
    assert result.probability_improvement is None


def test_paired_group_bootstrap_requires_matched_finite_groups():
    with pytest.raises(ValueError, match="common held-out group"):
        paired_group_bootstrap({"a": 1.0}, {"b": 1.0})
    with pytest.raises(ValueError, match="finite"):
        paired_group_bootstrap({"a": float("nan")}, {"a": 1.0})
