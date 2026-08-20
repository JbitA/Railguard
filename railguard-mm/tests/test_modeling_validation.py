import pandas as pd
import pytest

from ml.railguard_ml.contracts import DEPLOYMENT_SENSOR_COLUMNS
from ml.railguard_ml.validation import validate_modeling_table


def _frame():
    row = {name: 0.5 for name in DEPLOYMENT_SENSOR_COLUMNS}
    row.update({
        "vibration_rms": 1.0,
        "vibration_peak": 2.0,
        "vibration_kurtosis": 3.0,
        "crest_factor": 2.0,
        "vision_motion": 0.2,
        "humidity": 0.5,
        "speed_mps": 3.0,
        "image_path": "frame.jpg",
        "run_id": "run-a",
        "ts": "2026-01-01T00:00:00Z",
    })
    return pd.DataFrame([row])


def test_modeling_table_rejects_nonfinite_required_context():
    frame = _frame(); frame.loc[0, "temperature_c"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        validate_modeling_table(frame, DEPLOYMENT_SENSOR_COLUMNS)


def test_modeling_table_rejects_out_of_contract_physical_values():
    frame = _frame(); frame.loc[0, "humidity"] = 1.2
    with pytest.raises(ValueError, match="humidity"):
        validate_modeling_table(frame, DEPLOYMENT_SENSOR_COLUMNS)
    frame = _frame(); frame.loc[0, "vision_motion"] = -0.1
    with pytest.raises(ValueError, match="vision_motion"):
        validate_modeling_table(frame, DEPLOYMENT_SENSOR_COLUMNS)


def test_modeling_table_accepts_complete_finite_inputs():
    validate_modeling_table(_frame(), DEPLOYMENT_SENSOR_COLUMNS)
