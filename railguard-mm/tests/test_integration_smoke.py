from scripts.integration_smoke import SCHEMA, ack_matches_expected, build_record
from cloud.ingestor.validation import load_validator, validate_record


def test_integration_smoke_record_matches_production_schema():
    record = build_record("railguard-test")
    validate_record(load_validator(SCHEMA), record)
    assert record["prediction"]["horizons"] == [1, 5, 10]
    assert [x["sensor_id"] for x in record["vibration"]["sensors"]] == [0, 1, 2]


def test_integration_smoke_ack_must_match_exact_record_identity():
    record = build_record("railguard-test", seq=7)
    key = f"{record['device_id']}|{record['ts']}|{record['seq']}"
    valid = {
        "schema_version": 1,
        "device_id": record["device_id"],
        "ts": record["ts"],
        "seq": record["seq"],
        "ack_key": key,
    }
    import json
    assert ack_matches_expected(json.dumps(valid), record["device_id"], key)
    wrong = dict(valid, seq=8, ack_key=f"{record['device_id']}|{record['ts']}|8")
    assert not ack_matches_expected(json.dumps(wrong), record["device_id"], key)
    forged_key = dict(valid, ack_key="forged")
    assert not ack_matches_expected(json.dumps(forged_key), record["device_id"], key)


def test_integration_smoke_can_require_signed_exact_ack():
    from cloud.ingestor.ack import ack_payload
    record = build_record("railguard-test", seq=11)
    key = f"{record['device_id']}|{record['ts']}|{record['seq']}"
    signed = ack_payload(record, "smoke-secret")
    assert ack_matches_expected(signed, record["device_id"], key, "smoke-secret")
    assert not ack_matches_expected(signed, record["device_id"], key, "wrong")
