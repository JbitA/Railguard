from __future__ import annotations

from cloud.ingestor.ack import ack_payload as cloud_ack_payload, ack_topic as cloud_ack_topic
from edge.railguard_edge.ack import decode_ack, ack_topic_for_device


def test_edge_and_cloud_application_ack_contracts_are_compatible():
    record = {"device_id": "railguard-9", "ts": "2026-08-20T12:00:00Z", "seq": 99}
    assert cloud_ack_topic(record) == ack_topic_for_device(record["device_id"])
    key = decode_ack(cloud_ack_payload(record), record["device_id"])
    assert key.endswith("|99")


def test_cloud_signed_ack_is_verified_by_edge_codec():
    from cloud.ingestor.ack import ack_payload
    from edge.railguard_edge.ack import decode_ack
    record = {"device_id": "railguard-1", "ts": "2026-08-20T12:00:00Z", "seq": 9}
    payload = ack_payload(record, "shared-secret")
    assert decode_ack(payload, "railguard-1", "shared-secret") == "railguard-1|2026-08-20T12:00:00Z|9"
    import pytest
    with pytest.raises(ValueError, match="HMAC"):
        decode_ack(payload, "railguard-1", "wrong-secret")
