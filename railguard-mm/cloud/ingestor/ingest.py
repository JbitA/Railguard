from __future__ import annotations

import json
import os
import time

try:
    from .processor import process_record, topic_matches_device
    from .validation import load_validator
    from .ack import ack_payload, ack_topic
except ImportError:  # direct execution inside the ingestor container
    from processor import process_record, topic_matches_device
    from validation import load_validator
    from ack import ack_payload, ack_topic

DB = os.getenv("DATABASE_URL", "postgresql://railguard:railguard_dev@timescaledb:5432/railguard")
MQTT_HOST = os.getenv("MQTT_HOST", "mosquitto")
MQTT_PORT = int(os.getenv("MQTT_PORT", "1883"))
TOPIC = os.getenv("MQTT_TOPIC", "railguard/telemetry/#")
MQTT_CLIENT_ID = os.getenv("MQTT_CLIENT_ID", "railguard-timescale-ingestor-v1")
SCHEMA_PATH = os.getenv("TELEMETRY_SCHEMA", "/app/schema/telemetry.schema.json")
ACK_HMAC_KEY = os.getenv("ACK_HMAC_KEY") or None


def connect_db():
    import psycopg
    while True:
        try:
            return psycopg.connect(DB, autocommit=True)
        except Exception as exc:
            print("database unavailable:", exc, flush=True)
            time.sleep(2)


def mqtt_session_contract() -> dict:
    if not MQTT_CLIENT_ID:
        raise ValueError("MQTT_CLIENT_ID must be non-empty for a persistent session")
    return {"client_id": MQTT_CLIENT_ID, "clean_session": False, "qos": 1, "topic": TOPIC}


def build_mqtt_client(on_message):
    """Create a stable MQTT 3.1.1 persistent session for durable QoS-1 delivery."""
    import paho.mqtt.client as mqtt

    contract = mqtt_session_contract()
    client = mqtt.Client(
        mqtt.CallbackAPIVersion.VERSION2,
        client_id=contract["client_id"],
        clean_session=contract["clean_session"],
        protocol=mqtt.MQTTv311,
    )
    client.on_message = on_message
    client.reconnect_delay_set(min_delay=1, max_delay=30)

    def on_connect(client, userdata, flags, reason_code, properties):
        if reason_code != 0:
            print(f"MQTT connect failed: {reason_code}", flush=True)
            return
        # Re-subscribing is idempotent and recreates the subscription if the broker
        # lost session state, while clean_session=False preserves queued QoS-1 data
        # during ordinary ingestor downtime.
        client.subscribe(contract["topic"], qos=contract["qos"])
        print(f"subscribed to {TOPIC} client_id={MQTT_CLIENT_ID} persistent_session=true", flush=True)

    client.on_connect = on_connect
    return client


def process_mqtt_message(client, state: dict, validator, msg, *, reconnect_db=connect_db):
    """Validate, persist, then emit an application ACK.

    The ordering is deliberate: no ACK can be queued until process_record has
    completed the telemetry write and all prediction-horizon writes. Database
    failures reconnect/retry once; validation/data failures are never ACKed.
    """
    record = json.loads(
        msg.payload,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if not topic_matches_device(msg.topic, str(record.get("device_id", ""))):
        raise ValueError(
            f"MQTT topic/device_id mismatch: {msg.topic!r} vs {record.get('device_id')!r}"
        )
    try:
        process_record(state["conn"], validator, record)
    except Exception as exc:
        # Only reconnect when the DB operation failed. Schema/data errors must not
        # cause reconnection storms; validate once before retrying DB work.
        from jsonschema import ValidationError
        if isinstance(exc, (ValueError, ValidationError)):
            raise
        try:
            state["conn"].close()
        except Exception:
            pass
        state["conn"] = reconnect_db()
        process_record(state["conn"], validator, record)

    # Application-level ACK: queue only after the complete persistence path above
    # returned successfully. A lost ACK is harmless because edge retries are
    # idempotent under the same immutable identity.
    ack_info = client.publish(ack_topic(record), ack_payload(record, ACK_HMAC_KEY), qos=1)
    if ack_info.rc != 0:
        raise RuntimeError(f"failed to queue database ACK rc={ack_info.rc}")
    return record


def run():
    validator = load_validator(SCHEMA_PATH)
    state = {"conn": connect_db()}

    def on_message(client, userdata, msg):
        try:
            process_mqtt_message(client, state, validator, msg)
        except Exception as exc:
            print("ingest error:", exc, flush=True)

    client = build_mqtt_client(on_message)
    # connect() may fail during a broker restart. Keep the service alive and retry;
    # once connected, loop_forever handles subsequent reconnects.
    while True:
        try:
            client.connect(MQTT_HOST, MQTT_PORT, 30)
            break
        except OSError as exc:
            print("MQTT unavailable:", exc, flush=True)
            time.sleep(2)
    client.loop_forever(retry_first_connection=True)


if __name__ == "__main__":
    run()
