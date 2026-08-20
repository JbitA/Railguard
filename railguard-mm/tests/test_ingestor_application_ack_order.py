from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from cloud.ingestor.ingest import process_mqtt_message
from cloud.ingestor.validation import load_validator
from tests.test_ingestor_processor import sample_record

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = load_validator(ROOT / "schemas/telemetry.schema.json")


class _AckInfo:
    rc = 0


class _Client:
    def __init__(self, events):
        self.events = events
        self.published = []

    def publish(self, topic, payload, qos):
        self.events.append("ack")
        self.published.append((topic, payload, qos))
        return _AckInfo()


class _Conn:
    def __init__(self, events, *, fail_at=None):
        self.events = events
        self.fail_at = fail_at
        self.calls = 0
        self.closed = False

    def execute(self, sql, params):
        self.calls += 1
        self.events.append(f"db{self.calls}")
        if self.calls == self.fail_at:
            raise RuntimeError("injected database failure")

    def close(self):
        self.closed = True


def _msg(record):
    return SimpleNamespace(
        topic=f"railguard/telemetry/{record['device_id']}",
        payload=json.dumps(record).encode(),
    )


def test_commit_ack_is_queued_only_after_all_database_writes():
    events = []
    conn = _Conn(events)
    client = _Client(events)
    record = sample_record()
    process_mqtt_message(client, {"conn": conn}, VALIDATOR, _msg(record))
    # telemetry + 3 prediction horizons, then ACK
    assert events == ["db1", "db2", "db3", "db4", "ack"]
    assert len(client.published) == 1


def test_database_failure_does_not_emit_ack_when_retry_also_fails():
    events = []
    first = _Conn(events, fail_at=2)
    retry = _Conn(events, fail_at=1)
    client = _Client(events)
    state = {"conn": first}
    record = sample_record()
    with pytest.raises(RuntimeError, match="injected database failure"):
        process_mqtt_message(client, state, VALIDATOR, _msg(record), reconnect_db=lambda: retry)
    assert client.published == []
    assert "ack" not in events
    assert first.closed
