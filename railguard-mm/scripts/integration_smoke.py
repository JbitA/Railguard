from __future__ import annotations

import argparse
import json
import os
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


try:
    from cloud.ingestor.validation import load_validator, validate_record
except ModuleNotFoundError:
    import sys
    ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(ROOT))
    from cloud.ingestor.validation import load_validator, validate_record

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = ROOT / "schemas/telemetry.schema.json"

try:
    from edge.railguard_edge.ack import decode_ack
except ModuleNotFoundError:
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from edge.railguard_edge.ack import decode_ack


def ack_matches_expected(
    payload: bytes | str,
    expected_device_id: str,
    expected_ack_key: str,
    hmac_key: str | None = None,
) -> bool:
    try:
        return decode_ack(payload, expected_device_id, hmac_key) == expected_ack_key
    except (ValueError, json.JSONDecodeError, UnicodeDecodeError, TypeError):
        return False


def build_record(device_id: str = "railguard-ci-smoke", *, seq: int = 424242) -> dict:
    ts = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    sensors = [
        {
            "sensor_id": i,
            "rms_ms2": 1.2 + 0.1 * i,
            "peak_ms2": 3.5 + 0.1 * i,
            "kurtosis": 3.1,
            "crest_factor": 2.9,
            "band_energy": [0.1, 0.2, 0.3, 0.4],
        }
        for i in range(3)
    ]
    return {
        "schema_version": 1,
        "device_id": device_id,
        "ts": ts,
        "seq": seq,
        "sample_period_ms": 100.0,
        "gps": {"lat": 40.24, "lon": -77.90, "speed_mps": 7.2},
        "environment": {"temperature_c": 21.5, "humidity": 0.55},
        "vibration": {
            "rms_ms2": 1.3,
            "peak_ms2": 3.6,
            "kurtosis": 3.1,
            "crest_factor": 2.9,
            "band_energy": [0.1, 0.2, 0.3, 0.4],
            "sensors": sensors,
        },
        "vision": {"motion_score": 0.21, "contrast": 0.72, "sharpness": 1.4, "frame_ref": None},
        "health": {
            "packet_loss": 0,
            "spool_depth": 0,
            "spool_dropped": 0,
            "camera_matched": True,
            "sync_error_ms": 2.1,
            "sensor_skew_ms": 3.0,
            "clock_alignment_locked": True,
            "clock_jitter_ms": 1.2,
            "clock_samples": 64,
            "context_flags": 3,
        },
        "prediction": {
            "model_version": "integration-smoke-model",
            "horizons": [1, 5, 10],
            "step_ms": 100.0,
            "vibration_rms": [1.31, 1.35, 1.42],
            "vision_motion": [0.22, 0.24, 0.27],
            "anomaly_probability": 0.08,
        },
    }


def _json_get(url: str, timeout: float = 2.0):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def wait_http(api_base: str, deadline: float) -> None:
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if _json_get(f"{api_base}/health").get("status") == "ok":
                return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"API did not become healthy before timeout: {last_error}")


def connect_mqtt(
    host: str,
    port: int,
    deadline: float,
    *,
    ack_topic: str | None = None,
    ack_event: threading.Event | None = None,
    expected_ack_key: str | None = None,
    ack_hmac_key: str | None = None,
):
    import paho.mqtt.client as mqtt
    connected = threading.Event()
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=f"railguard-integration-smoke-{os.getpid()}",
        clean_session=True,
        protocol=mqtt.MQTTv311,
    )

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code == 0:
            if ack_topic:
                client.subscribe(ack_topic, qos=1)
            connected.set()

    def on_message(client, userdata, msg):
        if ack_event is not None and ack_topic is not None and msg.topic == ack_topic:
            if expected_ack_key is None:
                return
            expected_device_id = ack_topic.rsplit("/", 1)[-1]
            if ack_matches_expected(msg.payload, expected_device_id, expected_ack_key, ack_hmac_key):
                ack_event.set()

    client.on_connect = on_connect
    client.on_message = on_message
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            client.connect(host, port, keepalive=10)
            client.loop_start()
            if connected.wait(timeout=min(3.0, max(0.1, deadline - time.monotonic()))):
                return client
            client.loop_stop()
            client.disconnect()
        except OSError as exc:
            last_error = exc
        time.sleep(0.5)
    raise RuntimeError(f"MQTT broker did not become reachable before timeout: {last_error}")


def verify_api(api_base: str, record: dict) -> bool:
    latest = _json_get(f"{api_base}/v1/latest/{record['device_id']}")
    if latest.get("seq") != record["seq"] or latest.get("schema_version") != 1:
        return False
    if latest.get("sensor0_rms") is None or latest.get("context_flags") != 3:
        return False
    series = _json_get(f"{api_base}/v1/series/{record['device_id']}?minutes=5")
    predictions = [p for p in series.get("predicted", []) if p.get("model_version") == "integration-smoke-model"]
    horizons = {int(p["horizon_steps"]) for p in predictions}
    return horizons == {1, 5, 10}


def _dotenv_value(path: Path, key: str) -> str | None:
    if not path.exists():
        return None
    prefix = key + "="
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            return value or None
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Smoke-test MQTT -> ingestor -> TimescaleDB -> API with one schema-valid record.")
    ap.add_argument("--api-base", default="http://127.0.0.1:8000")
    ap.add_argument("--mqtt-host", default="127.0.0.1")
    ap.add_argument("--mqtt-port", type=int, default=1883)
    ap.add_argument("--device-id", default="railguard-ci-smoke")
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument(
        "--ack-hmac-key",
        default=os.getenv("ACK_HMAC_KEY") or _dotenv_value(ROOT / ".env", "ACK_HMAC_KEY"),
        help="shared post-commit ACK HMAC key; defaults to ACK_HMAC_KEY or local .env",
    )
    args = ap.parse_args()

    record = build_record(args.device_id)
    validator = load_validator(SCHEMA)
    validate_record(validator, record)
    deadline = time.monotonic() + args.timeout
    wait_http(args.api_base.rstrip("/"), deadline)
    ack_event = threading.Event()
    ack_topic = f"railguard/ack/{record['device_id']}"
    expected_ack_key = f"{record['device_id']}|{record['ts']}|{record['seq']}"
    client = connect_mqtt(
        args.mqtt_host,
        args.mqtt_port,
        deadline,
        ack_topic=ack_topic,
        ack_event=ack_event,
        expected_ack_key=expected_ack_key,
        ack_hmac_key=args.ack_hmac_key,
    )
    topic = f"railguard/telemetry/{record['device_id']}"
    payload = json.dumps(record, allow_nan=False, separators=(",", ":"))
    try:
        # Re-publish idempotently until the DB/API observes the record. This makes the
        # smoke test robust to the ingestor's first subscription racing the publisher.
        while time.monotonic() < deadline:
            info = client.publish(topic, payload, qos=1, retain=False)
            info.wait_for_publish(timeout=3.0)
            time.sleep(0.75)
            try:
                if verify_api(args.api_base.rstrip("/"), record) and ack_event.is_set():
                    print("integration smoke: PASS (MQTT -> TimescaleDB commit ACK -> API + 3 prediction horizons)")
                    return 0
            except (OSError, urllib.error.URLError, json.JSONDecodeError):
                pass
        raise RuntimeError("record/predictions did not become visible through the API before timeout")
    finally:
        client.loop_stop()
        client.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
