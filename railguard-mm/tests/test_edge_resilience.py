from datetime import datetime, timezone
from edge.railguard_edge.serial_protocol import Packet, packet_timestamp_iso
from edge.railguard_edge.native_bridge import parse_native_record
import pytest


def test_pps_timestamp_conversion():
    p = Packet(1, 1, 10, 1_700_000_000, 125_000, b"")
    assert packet_timestamp_iso(p) == "2023-11-14T22:13:20.125000Z"


def test_pps_timestamp_fallback():
    fallback = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
    p = Packet(1, 1, 10, 0, 10, b"")
    assert packet_timestamp_iso(p, fallback) == "2026-08-20T12:00:00.000000Z"


def test_native_bridge_rejects_nonfinite_and_nonobject_json():
    assert parse_native_record('{"schema_version":1,"x":1}')['x']==1
    with pytest.raises(ValueError): parse_native_record('{"x":NaN}')
    with pytest.raises(ValueError): parse_native_record('[1,2,3]')


def test_native_record_identity_can_be_checked_before_durable_publish():
    record = parse_native_record('{"device_id":"railguard-01"}')
    assert record["device_id"] == "railguard-01"
