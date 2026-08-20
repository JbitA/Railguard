import copy
import re
from pathlib import Path

import pytest

from cloud.ingestor.processor import process_record
from cloud.ingestor.validation import load_validator


def sample_record():
    return {
      "schema_version":1,"device_id":"railguard-001","ts":"2026-08-20T12:00:00Z","seq":1,"sample_period_ms":100.0,
      "gps":{"lat":40.0,"lon":-77.0,"speed_mps":6.0},
      "environment":{"temperature_c":20.0,"humidity":0.5},
      "vibration":{"rms_ms2":2.0,"peak_ms2":5.0,"kurtosis":3.0,"crest_factor":2.5,"band_energy":[.1,.2,.3,.4],
                   "sensors":[
                     {"sensor_id":0,"rms_ms2":1.8,"peak_ms2":4.0,"kurtosis":3.0,"crest_factor":2.2,"band_energy":[.1,.2,.3,.4]},
                     {"sensor_id":1,"rms_ms2":2.0,"peak_ms2":5.0,"kurtosis":3.1,"crest_factor":2.5,"band_energy":[.1,.2,.3,.4]},
                     {"sensor_id":2,"rms_ms2":2.2,"peak_ms2":5.5,"kurtosis":3.2,"crest_factor":2.5,"band_energy":[.1,.2,.3,.4]}]},
      "vision":{"motion_score":.2,"contrast":.7,"sharpness":1.2,"frame_ref":None},
      "health":{"packet_loss":0,"spool_depth":0,"camera_matched":True,"sync_error_ms":1.0,"sensor_skew_ms":3.0,
                "clock_alignment_locked":True,"clock_jitter_ms":2.0,"clock_samples":16},
      "prediction":{"model_version":"test","horizons":[1,5,10],"step_ms":100.0,"vibration_rms":[2.1,2.2,2.3],"vision_motion":[.2,.2,.2],"anomaly_probability":.1}
    }

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = load_validator(ROOT / "schemas/telemetry.schema.json")


class FakeConnection:
    def __init__(self):
        self.calls = []

    def execute(self, sql, params):
        self.calls.append((sql, params))


def test_processor_validates_maps_and_fans_out_three_prediction_horizons():
    conn = FakeConnection()
    row, prediction_rows = process_record(conn, VALIDATOR, sample_record())
    assert row["schema_version"] == 1
    assert prediction_rows == 3
    assert len(conn.calls) == 4
    assert conn.calls[0][1]["sensor2_rms"] == 2.2
    target_offsets = [call[1][2] for call in conn.calls[1:]]
    assert target_offsets == [100.0, 500.0, 1000.0]
    assert all(call[1][4] == 1 for call in conn.calls[1:])  # source_seq


def test_processor_rejects_invalid_contract_before_any_database_write():
    conn = FakeConnection()
    bad = copy.deepcopy(sample_record())
    bad["schema_version"] = 2
    with pytest.raises(ValueError):
        process_record(conn, VALIDATOR, bad)
    assert conn.calls == []


def test_insert_named_placeholders_are_all_supplied_by_flattened_row():
    from cloud.ingestor.processor import INSERT
    from cloud.ingestor.transform import flatten

    row = flatten(sample_record())
    placeholders = set(re.findall(r"%\(([^)]+)\)s", INSERT))
    assert placeholders == set(row)


def test_database_idempotency_key_matches_application_ack_identity():
    from cloud.ingestor.processor import INSERT, PREDICTION_INSERT
    init = (ROOT / "cloud/db/init.sql").read_text()
    assert "PRIMARY KEY (device_id, ts, seq)" in init
    assert "ON CONFLICT (device_id, ts, seq)" in INSERT
    assert "source_seq BIGINT NOT NULL" in init
    assert "PRIMARY KEY (device_id, issued_at, source_seq, target_ts, model_version)" in init
    assert "ON CONFLICT (device_id, issued_at, source_seq, target_ts, model_version)" in PREDICTION_INSERT
