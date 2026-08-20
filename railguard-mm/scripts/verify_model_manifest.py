#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml.railguard_ml.contracts import DEPLOYMENT_IMAGE_MODE, DEPLOYMENT_SENSOR_COLUMNS, MODEL_ARCH_VERSION

MODEL_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_contract(manifest: dict) -> dict:
    columns = list(manifest.get("sensor_columns") or [])
    if columns != DEPLOYMENT_SENSOR_COLUMNS:
        raise ValueError("deployment manifest sensor_columns do not match native runtime feature order")
    if manifest.get("image_mode") != DEPLOYMENT_IMAGE_MODE:
        raise ValueError("deployment manifest image_mode does not match native monochrome camera contract")
    sequence_length = int(manifest.get("sequence_length", 0))
    if sequence_length < 4 or sequence_length > 128:
        raise ValueError("deployment manifest sequence_length must be in [4,128]")
    sample_period_ms = float(manifest.get("sample_period_ms", 0.0))
    if not (0.0 < sample_period_ms <= 10000.0):
        raise ValueError("deployment manifest sample_period_ms is invalid")
    if list(manifest.get("forecast_horizons_steps") or []) != [1, 5, 10]:
        raise ValueError("deployment manifest forecast horizon contract is incompatible")
    return {
        "model_version": str(manifest["model_version"]),
        "sequence_length": sequence_length,
        "sample_period_ms": sample_period_ms,
        "image_mode": DEPLOYMENT_IMAGE_MODE,
        "sensor_columns": list(DEPLOYMENT_SENSOR_COLUMNS),
    }


def _load_common(manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("manifest_version", 0)) != 1:
        raise ValueError("unsupported deployment manifest_version")
    if int(manifest.get("model_arch_version", -1)) != MODEL_ARCH_VERSION:
        raise ValueError(
            f"deployment model_arch_version={manifest.get('model_arch_version')} does not match runtime contract {MODEL_ARCH_VERSION}"
        )
    version = str(manifest.get("model_version", ""))
    if not MODEL_VERSION_RE.fullmatch(version):
        raise ValueError("invalid or missing model_version in deployment manifest")
    checkpoint_sha = str(manifest.get("checkpoint_sha256", "")).lower()
    if not SHA256_RE.fullmatch(checkpoint_sha):
        raise ValueError("deployment manifest has no valid checkpoint_sha256")
    if version != f"fusion-{checkpoint_sha[:12]}":
        raise ValueError("model_version is not derived from checkpoint_sha256")
    runtime_contract(manifest)
    return manifest


def verify_onnx_manifest(onnx: Path, manifest_path: Path) -> dict:
    manifest = _load_common(manifest_path)
    expected = str(manifest.get("onnx_sha256", "")).lower()
    if not SHA256_RE.fullmatch(expected):
        raise ValueError("deployment manifest has no valid onnx_sha256")
    actual = sha256(onnx)
    if actual != expected:
        raise ValueError(f"ONNX SHA-256 mismatch: expected {expected}, got {actual}")
    return runtime_contract(manifest)


def verify_engine_manifest(engine: Path, manifest_path: Path) -> dict:
    manifest = _load_common(manifest_path)
    expected_onnx = str(manifest.get("onnx_sha256", "")).lower()
    if not SHA256_RE.fullmatch(expected_onnx):
        raise ValueError("deployment manifest has no valid onnx_sha256 lineage")
    expected = str(manifest.get("engine_sha256", "")).lower()
    if not SHA256_RE.fullmatch(expected):
        raise ValueError("deployment manifest has no valid engine_sha256")
    actual = sha256(engine)
    if actual != expected:
        raise ValueError(f"TensorRT engine SHA-256 mismatch: expected {expected}, got {actual}")
    return runtime_contract(manifest)


def emit_contract(contract: dict, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(contract, separators=(",", ":")))
    elif fmt == "tsv":
        print(f"{contract['model_version']}\t{contract['sequence_length']}\t{contract['sample_period_ms']:.17g}")
    else:
        print(contract["model_version"])


def main() -> None:
    p = argparse.ArgumentParser(description="Verify RailGuard ONNX/TensorRT artifacts against deployment lineage and runtime contract")
    p.add_argument("artifact", type=Path)
    p.add_argument("--manifest", type=Path, default=None)
    p.add_argument("--kind", choices=["engine", "onnx"], default="engine")
    p.add_argument("--format", choices=["model-version", "tsv", "json"], default="model-version")
    a = p.parse_args()
    manifest = a.manifest or Path(str(a.artifact) + ".manifest.json")
    if a.kind == "onnx":
        contract = verify_onnx_manifest(a.artifact, manifest)
    else:
        contract = verify_engine_manifest(a.artifact, manifest)
    emit_contract(contract, a.format)


if __name__ == "__main__":
    main()
