from pathlib import Path

from cloud.ingestor import ingest


def test_ingestor_uses_stable_persistent_mqtt_session(monkeypatch):
    monkeypatch.setattr(ingest, "MQTT_CLIENT_ID", "railguard-test-ingestor")
    contract = ingest.mqtt_session_contract()
    assert contract["client_id"] == "railguard-test-ingestor"
    assert contract["clean_session"] is False
    assert contract["qos"] == 1
    assert contract["topic"].endswith("/#")


def test_broker_configuration_persists_session_queue():
    root = Path(__file__).resolve().parents[1]
    config = (root / "deploy/mosquitto.conf").read_text()
    compose = (root / "docker-compose.yml").read_text()
    assert "persistence true" in config
    assert "persistence_location /mosquitto/data/" in config
    assert "mosquitto-data:/mosquitto/data" in compose
