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


def ack_topic_for_device(device_id: str) -> str:
    if not device_id or "/" in device_id:
        raise ValueError("invalid device_id for ACK topic")
    return f"railguard/ack/{device_id}"


def encode_ack(record: dict, hmac_key: str | bytes | None = None) -> str:
    payload = {
        "schema_version": 1,
        "device_id": str(record["device_id"]),
        "ts": str(record["ts"]),
        "seq": int(record["seq"]),
    }
    payload["ack_key"] = telemetry_ack_key(payload["device_id"], payload["ts"], payload["seq"])
    if hmac_key is not None:
        payload["hmac_sha256"] = ack_signature(payload["ack_key"], hmac_key)
    return json.dumps(payload, separators=(",", ":"), allow_nan=False)


def decode_ack(payload: bytes | str, expected_device_id: str, hmac_key: str | bytes | None = None) -> str:
    if isinstance(payload, bytes):
        payload = payload.decode("utf-8")
    obj = json.loads(payload, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    if not isinstance(obj, dict) or obj.get("schema_version") != 1:
        raise ValueError("unsupported ACK payload")
    device_id = str(obj.get("device_id", ""))
    if device_id != expected_device_id:
        raise ValueError("ACK device_id mismatch")
    key = telemetry_ack_key(device_id, str(obj.get("ts", "")), int(obj.get("seq", -1)))
    if obj.get("ack_key") != key:
        raise ValueError("ACK key mismatch")
    if hmac_key is not None:
        supplied = str(obj.get("hmac_sha256", ""))
        expected = ack_signature(key, hmac_key)
        if not hmac.compare_digest(supplied, expected):
            raise ValueError("ACK HMAC mismatch")
    return key
