from __future__ import annotations

from types import SimpleNamespace

from edge.railguard_edge.ack import encode_ack
from edge.railguard_edge.config import MqttConfig
from edge.railguard_edge.publisher import TelemetryPublisher


class PublishInfo:
    rc = 0
    def wait_for_publish(self, timeout=None): return None
    def is_published(self): return True


class FakeClient:
    def __init__(self):
        self.published = []
        self.subscriptions = []
        self.on_connect = None; self.on_disconnect = None; self.on_message = None
    def username_pw_set(self, *args): pass
    def connect_async(self, *args, **kwargs): pass
    def loop_start(self): pass
    def loop_stop(self): pass
    def disconnect(self): pass
    def subscribe(self, topic, qos): self.subscriptions.append((topic, qos)); return (0, 1)
    def publish(self, topic, payload, qos):
        self.published.append((topic, payload, qos)); return PublishInfo()


def _record():
    return {
        "device_id": "railguard-1", "ts": "2026-08-20T12:00:00Z", "seq": 1,
        "health": {},
    }


def test_publisher_does_not_delete_on_broker_publish_and_deletes_on_db_ack(tmp_path):
    client = FakeClient()
    cfg = MqttConfig(topic="railguard/telemetry/railguard-1")
    publisher = TelemetryPublisher(cfg, tmp_path / "spool.sqlite", client=client, ack_retry_s=10.0)
    try:
        publisher.connected = True
        publisher.publish(_record())
        assert publisher.spool_depth() == 1
        assert publisher.flush() == 1
        assert publisher.spool_depth() == 1
        msg = SimpleNamespace(topic="railguard/ack/railguard-1", payload=encode_ack(_record()).encode())
        publisher._on_message(client, None, msg)
        assert publisher.spool_depth() == 0
    finally:
        publisher.close()


def test_publisher_subscribes_to_device_specific_commit_ack(tmp_path):
    client = FakeClient(); cfg = MqttConfig(topic="railguard/telemetry/railguard-1")
    publisher = TelemetryPublisher(cfg, tmp_path / "spool.sqlite", client=client)
    try:
        publisher._on_connect(client, None, None, 0, None)
        assert ("railguard/ack/railguard-1", 1) in client.subscriptions
    finally:
        publisher.close()


def test_publisher_requires_valid_hmac_when_configured(tmp_path):
    client = FakeClient()
    cfg = MqttConfig(topic="railguard/telemetry/railguard-1", ack_hmac_key="shared-secret")
    publisher = TelemetryPublisher(cfg, tmp_path / "signed.sqlite", client=client)
    try:
        publisher.publish(_record())
        unsigned = SimpleNamespace(topic="railguard/ack/railguard-1", payload=encode_ack(_record()).encode())
        publisher._on_message(client, None, unsigned)
        assert publisher.spool_depth() == 1
        signed = SimpleNamespace(topic="railguard/ack/railguard-1", payload=encode_ack(_record(), "shared-secret").encode())
        publisher._on_message(client, None, signed)
        assert publisher.spool_depth() == 0
    finally:
        publisher.close()
