from cloud.api.app.auth_core import write_key_valid
from cloud.ingestor.processor import topic_matches_device


def test_write_key_is_optional_only_when_not_configured():
    assert write_key_valid(None,None)
    assert write_key_valid("x",None)
    assert write_key_valid("secret","secret")
    assert not write_key_valid(None,"secret")
    assert not write_key_valid("wrong","secret")


def test_mqtt_topic_must_match_payload_device_identity():
    assert topic_matches_device("railguard/telemetry/railguard-001","railguard-001")
    assert not topic_matches_device("railguard/telemetry/railguard-002","railguard-001")
    assert not topic_matches_device("other/railguard-001","railguard-001")
    assert not topic_matches_device("railguard/telemetry/a/b","a/b")
