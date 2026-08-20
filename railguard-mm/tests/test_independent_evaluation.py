from __future__ import annotations

from ml.evaluate_independent_models import _markdown, paired_uncertainty


def _metric(vib: float, vis: float) -> dict:
    return {
        "vibration_mae": [vib, vib, vib],
        "vision_mae": [vis, vis, vis],
        "vibration_mae_mean": vib,
        "vision_mae_mean": vis,
    }


def test_paired_uncertainty_resamples_whole_heldout_groups():
    per_group = {
        str(i): {
            "persistence": _metric(4.0 + i, 0.5),
            "sensor_only": _metric(3.0 + i, 0.4),
            "vision_only": _metric(3.5 + i, 0.3),
            "multimodal": _metric(2.0 + i, 0.2),
            "windows": 10,
        }
        for i in range(4)
    }
    stats = paired_uncertainty(per_group, repeats=500, seed=5)
    assert stats["sensor_only"]["vibration_mae_mean"]["estimate"] == 1.0
    assert stats["sensor_only"]["vibration_mae_mean"]["probability_improvement"] == 1.0


def test_independent_markdown_labels_block_bootstrap_not_iid_windows():
    metric = _metric(1.0, 0.1)
    report = {
        "contract": {"split_mode": "spatial", "test_groups": ["a", "b", "c"]},
        "windows": 30,
        "metrics": {name: dict(metric) for name in ("persistence", "sensor_only", "vision_only", "multimodal")},
        "paired_group_uncertainty": {
            name: {
                "vibration_mae_mean": {"estimate": 0.1, "ci95_low": 0.01, "ci95_high": 0.2, "probability_improvement": 0.98},
                "vision_mae_mean": {"estimate": 0.01, "ci95_low": -0.01, "ci95_high": 0.02, "probability_improvement": 0.7},
            }
            for name in ("persistence", "sensor_only", "vision_only")
        },
        "bootstrap": {"repeats": 5000, "seed": 17},
    }
    text = _markdown(report)
    assert "whole held-out geographic blocks/runs" in text
    assert "not IID-window" in text
    assert "98.0%" in text
