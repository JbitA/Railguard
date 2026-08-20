from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


def telemetry_topic_for_device(device_id: str) -> str:
    return f"railguard/telemetry/{device_id}"

@dataclass
class MqttConfig:
    host: str = "localhost"
    port: int = 1883
    topic: str = "railguard/telemetry/railguard-001"
    qos: int = 1
    username: str | None = None
    password: str | None = None
    ack_hmac_key: str | None = None

@dataclass
class EdgeConfig:
    device_id: str
    mqtt: MqttConfig
    spool_path: str = "data/edge_spool.sqlite"
    spool_max_records: int = 200_000


def load_config(path: str | Path) -> EdgeConfig:
    raw = yaml.safe_load(Path(path).read_text())
    device_id = str(raw["device_id"])
    mqtt_cfg = MqttConfig(**raw.get("mqtt", {}))
    expected_topic = telemetry_topic_for_device(device_id)
    if mqtt_cfg.topic != expected_topic:
        raise ValueError(
            f"mqtt.topic must equal {expected_topic!r} for device_id={device_id!r}; got {mqtt_cfg.topic!r}"
        )
    if mqtt_cfg.qos != 1:
        raise ValueError("mqtt.qos must be 1 for the durable telemetry handoff contract")
    if not 1 <= int(mqtt_cfg.port) <= 65535:
        raise ValueError("mqtt.port must be in [1,65535]")
    return EdgeConfig(
        device_id=device_id,
        mqtt=mqtt_cfg,
        spool_path=raw.get("spool", {}).get("path", "data/edge_spool.sqlite"),
        spool_max_records=int(raw.get("spool", {}).get("max_records", 200_000)),
    )
