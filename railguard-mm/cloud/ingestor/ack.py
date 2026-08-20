from __future__ import annotations

import hashlib
import hmac
import json


def telemetry_ack_key(device_id: str, ts: str, seq: int) -> str:
    if not device_id or not ts or int(seq) < 0:
        raise ValueError("invalid telemetry ACK identity")
    return f"{device_id}|{ts}|{int(seq)}"


def ack_signature(ack_key: str, hmac_key: str | bytes) -> str:
    key = hmac_key.encode("utf-8") if isinstance(hmac_key, str) else bytes(hmac_key)
    if not key:
        raise ValueError("ACK HMAC key must be non-empty")
    return hmac.new(key, ack_key.encode("utf-8"), hashlib.sha256).hexdigest()


def ack_topic(record: dict) -> str:
    device_id = str(record["device_id"])
    if not device_id or "/" in device_id:
        raise ValueError("invalid device_id for ACK topic")
    return f"railguard/ack/{device_id}"


def ack_payload(record: dict, hmac_key: str | bytes | None = None) -> str:
    device_id = str(record["device_id"])
    ts = str(record["ts"])
    seq = int(record["seq"])
    key = telemetry_ack_key(device_id, ts, seq)
    payload = {
        "schema_version": 1,
        "device_id": device_id,
        "ts": ts,
        "seq": seq,
        "ack_key": key,
    }
    if hmac_key is not None:
        payload["hmac_sha256"] = ack_signature(key, hmac_key)
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)
