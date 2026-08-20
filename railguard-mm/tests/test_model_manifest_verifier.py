import hashlib
import json

import pytest

from ml.railguard_ml.contracts import DEPLOYMENT_IMAGE_MODE, DEPLOYMENT_SENSOR_COLUMNS, MODEL_ARCH_VERSION
from scripts.verify_model_manifest import verify_engine_manifest, verify_onnx_manifest


def _base(payload=b"checkpoint"):
    checkpoint_sha = hashlib.sha256(payload).hexdigest()
    return {
        "manifest_version": 1,
        "model_arch_version": MODEL_ARCH_VERSION,
        "model_version": f"fusion-{checkpoint_sha[:12]}",
        "checkpoint_sha256": checkpoint_sha,
        "sensor_columns": list(DEPLOYMENT_SENSOR_COLUMNS),
        "image_mode": DEPLOYMENT_IMAGE_MODE,
        "sequence_length": 32,
        "sample_period_ms": 100.0,
        "forecast_horizons_steps": [1, 5, 10],
    }


def _write_engine(tmp_path, payload=b"engine"):
    engine=tmp_path/"model.engine"; engine.write_bytes(payload)
    manifest=tmp_path/"model.engine.manifest.json"
    data=_base(); data.update({
        "onnx_sha256": hashlib.sha256(b"onnx").hexdigest(),
        "engine_sha256": hashlib.sha256(payload).hexdigest(),
    })
    manifest.write_text(json.dumps(data))
    return engine,manifest


def test_verified_engine_manifest_returns_bound_model_version(tmp_path):
    engine,manifest=_write_engine(tmp_path)
    data=json.loads(manifest.read_text())
    assert verify_engine_manifest(engine,manifest)["model_version"]==data["model_version"]


def test_verified_engine_manifest_rejects_tampered_engine(tmp_path):
    engine,manifest=_write_engine(tmp_path)
    engine.write_bytes(b"tampered")
    with pytest.raises(ValueError,match="SHA-256 mismatch"):
        verify_engine_manifest(engine,manifest)


def test_verified_engine_manifest_rejects_unsafe_or_unbound_model_version(tmp_path):
    engine,manifest=_write_engine(tmp_path)
    data=json.loads(manifest.read_text()); data["model_version"]="bad version with spaces"; manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="model_version"):
        verify_engine_manifest(engine,manifest)
    data=_base(); data.update({"model_version":"fusion-deadbeef0000","onnx_sha256":hashlib.sha256(b"onnx").hexdigest(),"engine_sha256":hashlib.sha256(b"engine").hexdigest()}); manifest.write_text(json.dumps(data)); engine.write_bytes(b"engine")
    with pytest.raises(ValueError,match="not derived"):
        verify_engine_manifest(engine,manifest)


def test_manifest_rejects_wrong_architecture_or_missing_checkpoint_lineage(tmp_path):
    engine,manifest=_write_engine(tmp_path)
    data=json.loads(manifest.read_text()); data["model_arch_version"]=MODEL_ARCH_VERSION-1; manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="model_arch_version"):
        verify_engine_manifest(engine,manifest)
    data=json.loads(manifest.read_text()); data["model_arch_version"]=MODEL_ARCH_VERSION; data.pop("checkpoint_sha256"); manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="checkpoint_sha256"):
        verify_engine_manifest(engine,manifest)


def test_onnx_manifest_binds_exact_export_bytes(tmp_path):
    onnx=tmp_path/"model.onnx"; onnx.write_bytes(b"onnx")
    manifest=tmp_path/"model.onnx.manifest.json"
    data=_base(); data["onnx_sha256"]=hashlib.sha256(b"onnx").hexdigest(); manifest.write_text(json.dumps(data))
    assert verify_onnx_manifest(onnx,manifest)["model_version"]==data["model_version"]
    onnx.write_bytes(b"tampered")
    with pytest.raises(ValueError,match="ONNX SHA-256 mismatch"):
        verify_onnx_manifest(onnx,manifest)


def test_manifest_rejects_runtime_contract_mismatch(tmp_path):
    engine, manifest = _write_engine(tmp_path)
    data = json.loads(manifest.read_text())
    data["image_mode"] = "rgb"
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="image_mode"):
        verify_engine_manifest(engine, manifest)

    data = _base()
    data.update({
        "onnx_sha256": hashlib.sha256(b"onnx").hexdigest(),
        "engine_sha256": hashlib.sha256(b"engine").hexdigest(),
    })
    data["sequence_length"] = 0
    manifest.write_text(json.dumps(data)); engine.write_bytes(b"engine")
    with pytest.raises(ValueError, match="sequence_length"):
        verify_engine_manifest(engine, manifest)
