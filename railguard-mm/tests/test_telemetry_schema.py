import json
from pathlib import Path
import pytest

from cloud.ingestor.validation import load_validator, validate_record

ROOT=Path(__file__).resolve().parents[1]
VALIDATOR=load_validator(ROOT/'schemas/telemetry.schema.json')


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


def test_current_native_contract_validates():
    validate_record(VALIDATOR,sample_record())


def test_schema_rejects_missing_timebase_and_invalid_humidity():
    record=sample_record(); record.pop('sample_period_ms')
    with pytest.raises(ValueError): validate_record(VALIDATOR,record)
    record=sample_record(); record['environment']['humidity']=1.5
    with pytest.raises(ValueError): validate_record(VALIDATOR,record)


def test_schema_enforces_timestamp_format_and_three_unique_sensor_ids():
    record=sample_record(); record['ts']='not-a-timestamp'
    with pytest.raises(ValueError): validate_record(VALIDATOR,record)
    record=sample_record(); record['vibration']['sensors'][2]['sensor_id']=1
    with pytest.raises(ValueError): validate_record(VALIDATOR,record)

def test_schema_rejects_negative_health_counters():
    record=sample_record(); record['health']['packet_loss']=-1
    with pytest.raises(ValueError): validate_record(VALIDATOR,record)
    record=sample_record(); record['health']['spool_depth']=-1
    with pytest.raises(ValueError): validate_record(VALIDATOR,record)


def test_schema_allows_explicit_null_context_but_rejects_bad_device_id():
    record=sample_record()
    record["gps"]={"lat":None,"lon":None,"speed_mps":None}
    record["environment"]={"temperature_c":None,"humidity":None}
    record["health"]["context_flags"]=0
    validate_record(VALIDATOR,record)
    record=sample_record(); record["device_id"]='bad/device'
    with pytest.raises(ValueError): validate_record(VALIDATOR,record)


def test_schema_rejects_physically_impossible_measurements_and_predictions():
    record = sample_record(); record["vibration"]["rms_ms2"] = -0.1
    with pytest.raises(ValueError): validate_record(VALIDATOR, record)
    record = sample_record(); record["vision"]["motion_score"] = 1.01
    with pytest.raises(ValueError): validate_record(VALIDATOR, record)
    record = sample_record(); record["prediction"]["vision_motion"][2] = 1.2
    with pytest.raises(ValueError): validate_record(VALIDATOR, record)


def test_schema_requires_exact_ordered_forecast_horizon_contract():
    record = sample_record(); record["prediction"]["horizons"] = [1, 10, 5]
    with pytest.raises(ValueError): validate_record(VALIDATOR, record)
    record = sample_record(); record["prediction"]["horizons"] = [1, 5, 5]
    with pytest.raises(ValueError): validate_record(VALIDATOR, record)


def test_timestamp_identity_requires_canonical_utc_z_form():
    import copy
    from scripts.integration_smoke import build_record
    from cloud.ingestor.validation import load_validator, validate_record
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    validator = load_validator(root / "schemas/telemetry.schema.json")
    good = build_record("railguard-test")
    validate_record(validator, good)
    bad = copy.deepcopy(good)
    bad["ts"] = good["ts"].replace("Z", "+00:00")
    import pytest
    with pytest.raises(Exception):
        validate_record(validator, bad)
