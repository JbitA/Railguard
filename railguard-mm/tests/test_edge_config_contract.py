from pathlib import Path

import pytest

from edge.railguard_edge.config import load_config, telemetry_topic_for_device


def _write(tmp_path: Path, *, device="railguard-01", topic=None, qos=1):
    topic = topic or telemetry_topic_for_device(device)
    path = tmp_path / "edge.yaml"
    path.write_text(
        f"device_id: {device}\n"
        "mqtt:\n"
        "  host: localhost\n"
        "  port: 1883\n"
        f"  topic: {topic}\n"
        f"  qos: {qos}\n"
    )
    return path


def test_edge_config_binds_device_identity_to_mqtt_topic(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert cfg.mqtt.topic == "railguard/telemetry/railguard-01"
    with pytest.raises(ValueError, match="mqtt.topic"):
        load_config(_write(tmp_path, topic="railguard/telemetry/other-device"))


def test_edge_config_requires_qos1_for_durable_handoff(tmp_path):
    with pytest.raises(ValueError, match="qos"):
        load_config(_write(tmp_path, qos=0))
