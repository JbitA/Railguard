from __future__ import annotations

import copy
import pytest

from ml.summarize_seed_runs import summarize_reports


def _report(seed: int, mm: float) -> dict:
    contract = {
        "model_arch_version": 2,
        "training_protocol_version": 3,
        "selection_objective_name": "x",
        "sensor_columns": ["f"],
        "image_mode": "mono",
        "seq_len": 32,
        "sample_period_s": 0.1,
        "split_mode": "spatial",
        "split_seed": 7,
        "training_seed": seed,
        "train_groups": ["a"], "validation_groups": ["b"], "test_groups": ["c"],
        "spatial_block_m": 500.0, "spatial_purge_margin_m": 30.0,
        "dataset_fingerprint_sha256": "a" * 64,
    }
    def metric(v): return {"vibration_mae_mean": v, "vision_mae_mean": v / 10.0}
    return {"contract": contract, "metrics": {
        "persistence": metric(4.0), "sensor_only": metric(3.0),
        "vision_only": metric(3.5), "multimodal": metric(mm),
    }}


def test_seed_summary_reports_variability_and_paired_wins():
    summary = summarize_reports([_report(1, 2.0), _report(2, 2.2), _report(3, 1.8)])
    assert summary["seed_count"] == 3
    assert summary["model_variability"]["multimodal"]["vibration_mae_mean"]["mean"] == pytest.approx(2.0)
    assert summary["multimodal_error_reduction_by_seed"]["sensor_only"]["vibration_mae_mean"]["wins"] == 3


def test_seed_summary_rejects_duplicate_seeds_or_contract_drift():
    a = _report(1, 2.0)
    with pytest.raises(ValueError, match="unique"):
        summarize_reports([a, copy.deepcopy(a)])
    b = _report(2, 2.0); b["contract"]["test_groups"] = ["different"]
    with pytest.raises(ValueError, match="same dataset/split"):
        summarize_reports([a, b])
