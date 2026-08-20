from __future__ import annotations

import json

import pytest

from edge.railguard_edge.ack import ack_topic_for_device, decode_ack, encode_ack, telemetry_ack_key
from edge.railguard_edge.outbox import DurableOutbox


def _record(seq=7):
    return {"device_id": "railguard-1", "ts": "2026-08-20T12:00:00.000000Z", "seq": seq}


def test_application_ack_roundtrip_binds_device_timestamp_and_sequence():
    record = _record()
    key = telemetry_ack_key(record["device_id"], record["ts"], record["seq"])
    assert decode_ack(encode_ack(record), record["device_id"]) == key
    assert ack_topic_for_device(record["device_id"]) == "railguard/ack/railguard-1"
    with pytest.raises(ValueError, match="device_id mismatch"):
        decode_ack(encode_ack(record), "other")


def test_outbox_survives_broker_publish_until_database_ack(tmp_path):
    record = _record()
    payload = json.dumps(record)
    key = telemetry_ack_key(record["device_id"], record["ts"], record["seq"])
    outbox = DurableOutbox(tmp_path / "spool.sqlite", max_records=10)
    try:
        outbox.enqueue(payload, key)
        row_id, _ = outbox.ready_batch(10, retry_after_s=2.0, now=100.0)[0]
        outbox.mark_sent(row_id, sent_at=100.0)
        assert outbox.depth() == 1  # broker PUBACK must not delete it
        assert outbox.ready_batch(10, retry_after_s=2.0, now=101.0) == []
        assert outbox.ready_batch(10, retry_after_s=2.0, now=102.1)
        assert outbox.acknowledge(key) is True
        assert outbox.depth() == 0
    finally:
        outbox.close()


def test_duplicate_native_record_does_not_duplicate_durable_obligation(tmp_path):
    record = _record(); payload = json.dumps(record)
    outbox = DurableOutbox(tmp_path / "spool.sqlite", max_records=10)
    try:
        outbox.enqueue(payload); outbox.enqueue(payload)
        assert outbox.depth() == 1
    finally:
        outbox.close()
