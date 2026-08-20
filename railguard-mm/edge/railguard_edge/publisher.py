from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from .ack import ack_topic_for_device, decode_ack, telemetry_ack_key
from .config import MqttConfig
from .outbox import DurableOutbox


class TelemetryPublisher:
    """Durable producer boundary retained until the cloud confirms DB persistence.

    publish() commits to SQLite and returns immediately. MQTT PUBACK only marks a
    send attempt; the SQLite row is deleted after an application ACK emitted by the
    ingestor *after* its TimescaleDB commit. Lost ACKs therefore cause idempotent
    re-delivery rather than silent data loss.
    """

    def __init__(self, cfg: MqttConfig, spool_path: str | Path = "data/edge_spool.sqlite", max_records: int = 200_000, flush_batch: int = 500, ack_retry_s: float = 2.0, client=None):
        if max_records < 1 or ack_retry_s <= 0:
            raise ValueError("invalid publisher bounds")
        self.cfg = cfg
        self.max_records = int(max_records)
        self.flush_batch = int(flush_batch)
        self.ack_retry_s = float(ack_retry_s)
        self.device_id = cfg.topic.rsplit("/", 1)[-1]
        self.ack_topic = ack_topic_for_device(self.device_id)
        if client is None:
            import paho.mqtt.client as mqtt
            client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client = client
        if cfg.username:
            self.client.username_pw_set(cfg.username, cfg.password)
        self.connected = False
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message
        self.spool_path = Path(spool_path)
        self.outbox = DurableOutbox(self.spool_path, max_records=self.max_records)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._worker: threading.Thread | None = None

    def _on_connect(self, client, userdata, flags, reason_code, properties):
        self.connected = reason_code == 0
        if self.connected:
            client.subscribe(self.ack_topic, qos=1)
        self._wake.set()

    def _on_disconnect(self, client, userdata, disconnect_flags, reason_code, properties):
        self.connected = False

    def _on_message(self, client, userdata, msg):
        if msg.topic != self.ack_topic:
            return
        try:
            key = decode_ack(msg.payload, self.device_id, self.cfg.ack_hmac_key)
        except (ValueError, json.JSONDecodeError, UnicodeDecodeError):
            return
        if self.outbox.acknowledge(key):
            self._wake.set()

    def start(self):
        self.client.connect_async(self.cfg.host, self.cfg.port, keepalive=30)
        self.client.loop_start()
        if self._worker is None:
            self._worker = threading.Thread(target=self._drain_loop, name="railguard-mqtt-outbox", daemon=True)
            self._worker.start()

    def spool_depth(self) -> int:
        return self.outbox.depth()

    def dropped_records(self) -> int:
        return self.outbox.dropped_records()

    def _send(self, payload: str) -> bool:
        if not self.connected:
            return False
        try:
            info = self.client.publish(self.cfg.topic, payload, qos=self.cfg.qos)
            if int(info.rc) != 0:
                return False
            info.wait_for_publish(timeout=2)
            return info.is_published()
        except Exception:
            self.connected = False
            return False

    def flush(self, limit: int | None = None):
        if not self.connected:
            return 0
        limit = self.flush_batch if limit is None else int(limit)
        rows = self.outbox.ready_batch(limit, retry_after_s=self.ack_retry_s)
        sent = 0
        for row_id, payload in rows:
            if not self._send(payload):
                break
            self.outbox.mark_sent(row_id)
            sent += 1
        return sent

    def _drain_loop(self):
        while not self._stop.is_set():
            if self.connected:
                sent = self.flush()
                if sent:
                    continue
            self._wake.wait(0.25)
            self._wake.clear()

    def publish(self, record: dict):
        health = record.setdefault("health", {})
        health["spool_depth"] = self.spool_depth()
        health["spool_dropped"] = self.dropped_records()
        payload = json.dumps(record, separators=(",", ":"), allow_nan=False)
        key = telemetry_ack_key(str(record["device_id"]), str(record["ts"]), int(record["seq"]))
        self.outbox.enqueue(payload, key)
        self._wake.set()

    def close(self):
        self._stop.set(); self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=3)
        try:
            self.client.loop_stop(); self.client.disconnect()
        finally:
            self.outbox.close()
