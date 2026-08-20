from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "benchmarks/results/latest.json"
START = "<!-- BENCHMARK_TABLE_START -->"
END = "<!-- BENCHMARK_TABLE_END -->"


def _table(data: dict) -> str:
    c = data["median"]["cpp20"]
    p = data["median"]["python"]
    cmp = data["comparison"]
    return "\n".join([
        START,
        "| Operation | Python reference | C++20 native | Improvement |",
        "|---|---:|---:|---:|",
        f"| Legacy CRC packet decode | {p['packet_decode_ns']:.0f} ns | {c['packet_decode_ns']:.0f} ns | **{cmp['packet_decode_speedup_cpp_vs_python']:.2f}x** |",
        f"| 94-byte production packet decode | {p['sensor_packet_decode_ns']:.0f} ns | {c['sensor_packet_decode_ns']:.0f} ns | **{cmp['sensor_packet_decode_speedup_cpp_vs_python']:.2f}x** |",
        f"| 94-byte production throughput | {p['sensor_packet_decode_mpps']:.3f} Mpacket/s | {c['sensor_packet_decode_mpps']:.3f} Mpacket/s | **{cmp['sensor_packet_decode_speedup_cpp_vs_python']:.2f}x** |",
        f"| Production decode + semantic/timestamp validation | {p['sensor_packet_accept_ns']:.0f} ns | {c['sensor_packet_accept_ns']:.0f} ns | **{cmp['sensor_packet_accept_speedup_cpp_vs_python']:.2f}x** |",
        f"| Validated production throughput | {p['sensor_packet_accept_mpps']:.3f} Mpacket/s | {c['sensor_packet_accept_mpps']:.3f} Mpacket/s | **{cmp['sensor_packet_accept_speedup_cpp_vs_python']:.2f}x** |",
        f"| 512-sample DSP window | {p['dsp_window_us']:.2f} us | {c['dsp_window_us']:.2f} us | **{cmp['dsp_speedup_cpp_vs_python']:.2f}x** |",
        END,
    ])


def replace_block(text: str, block: str) -> str:
    pattern = re.compile(re.escape(START) + r".*?" + re.escape(END), re.S)
    if not pattern.search(text):
        raise RuntimeError("benchmark markers missing")
    return pattern.sub(block, text, count=1)


def expected_files(data: dict) -> dict[Path, str]:
    block = _table(data)
    out = {}
    for rel in ("README.md", "docs/performance.md"):
        path = ROOT / rel
        out[path] = replace_block(path.read_text(), block)
    return out


def contract_checks(data: dict) -> list[str]:
    errors: list[str] = []
    protocol = (ROOT / "edge/cpp/include/railguard/protocol.hpp").read_text()
    m = re.search(r"kHeaderBytes\s*=\s*(\d+).*?kSensorFeaturePayloadBytes\s*=\s*(\d+)", protocol, re.S)
    if not m or int(m.group(1)) + int(m.group(2)) + 4 != 94:
        errors.append("C++ production packet is not 94 bytes")
    schema = json.loads((ROOT / "schemas/telemetry.schema.json").read_text())
    if schema.get("properties", {}).get("schema_version", {}).get("const") != 1:
        errors.append("telemetry schema_version const is not 1")
    from ml.railguard_ml.contracts import DEPLOYMENT_IMAGE_MODE, DEPLOYMENT_SENSOR_COLUMNS, MODEL_ARCH_VERSION
    from ml.railguard_ml.objectives import TRAINING_PROTOCOL_VERSION
    contract = (ROOT / "edge/cpp/include/railguard/model_contract.hpp").read_text()
    m = re.search(r"kSensorFeatureDim\s*=\s*(\d+)", contract)
    if not m or int(m.group(1)) != len(DEPLOYMENT_SENSOR_COLUMNS):
        errors.append("Python/C++ deployment feature dimensions disagree")
    if MODEL_ARCH_VERSION < 2:
        errors.append("model architecture version regressed below the physically bounded deployment contract")
    if TRAINING_PROTOCOL_VERSION < 3:
        errors.append("training protocol version regressed below joint-selection/monochrome contract")
    if f'kImageMode = "{DEPLOYMENT_IMAGE_MODE}"' not in contract:
        errors.append("Python/C++ deployment image modes disagree")
    compose = (ROOT / "docker-compose.yml").read_text()
    for stale in ("timescale/timescaledb:latest", "timescale/timescaledb:latest-pg16", "eclipse-mosquitto:2\n", "minio/minio:latest"):
        if stale in compose:
            errors.append(f"floating Compose image tag remains: {stale.strip()}")
    mosquitto = (ROOT / "deploy/mosquitto.conf").read_text()
    if "persistence true" not in mosquitto or "persistence_location /mosquitto/data/" not in mosquitto:
        errors.append("reference MQTT broker persistence contract is missing")
    ingest = (ROOT / "cloud/ingestor/ingest.py").read_text()
    if '"clean_session": False' not in ingest or '"qos": 1' not in ingest:
        errors.append("ingestor persistent QoS-1 MQTT session contract is missing")
    readme = (ROOT / "README.md").read_text()
    if "EDGE -->|TLS MQTT| MQTT" in readme:
        errors.append("README incorrectly claims the local MQTT edge is TLS")
    dockerfiles = {
        "cloud/ingestor/Dockerfile": "FROM python:3.12.13-slim-bookworm",
        "cloud/ml_worker/Dockerfile": "FROM python:3.12.13-slim-bookworm",
        "cloud/api/Dockerfile": "FROM python:3.12.13-slim-bookworm",
        "web/Dockerfile": "FROM node:22.23.2-alpine3.24 AS build",
    }
    for rel, expected in dockerfiles.items():
        if expected not in (ROOT / rel).read_text():
            errors.append(f"container runtime tag drift: {rel}")
    req = (ROOT / "requirements-dev.txt").read_text().splitlines()
    required_pkgs = ("pandas==", "scikit-learn==", "opencv-python-headless==", "huggingface-hub==", "PyYAML==")
    for prefix in required_pkgs:
        if not any(line.startswith(prefix) for line in req):
            errors.append(f"host-test dependency is undeclared: {prefix[:-2]}")
    docs = "\n".join((ROOT / p).read_text() for p in ["README.md", "docs/performance.md", "docs/verification.md"])
    for stale in ("90-byte", "1242 ns", "2.73x", "20.0x faster"):
        if stale in docs:
            errors.append(f"stale evidence token remains: {stale}")

    repeats = int(data.get("repeats", 0))
    performance = (ROOT / "docs/performance.md").read_text()
    verification = (ROOT / "docs/verification.md").read_text()
    if repeats < 1 or f"repeats: {repeats}; table values are medians" not in performance:
        errors.append("performance benchmark repeat count drifted from latest.json")
    if f"{repeats}-repeat benchmark in `benchmarks/results/latest.json`" not in verification:
        errors.append("verification benchmark repeat count drifted from latest.json")

    # Manifest-verifying launcher owns the temporal runtime contract. A documented
    # seq-len override would bypass/contradict that fail-closed deployment boundary.
    wrapper = re.search(
        r"deploy/tensorrt/run_edge_verified\.sh.*?```",
        readme,
        re.S,
    )
    if not wrapper or "--seq-len" in wrapper.group(0) or "--model-step-ms" in wrapper.group(0):
        errors.append("README verified-launch example overrides or omits manifest-controlled runtime contract")

    publisher = (ROOT / "edge/railguard_edge/publisher.py").read_text()
    cloud_ingest = (ROOT / "cloud/ingestor/ingest.py").read_text()
    smoke = (ROOT / "scripts/integration_smoke.py").read_text()
    if "outbox.acknowledge" not in publisher or "ack_retry_s" not in publisher:
        errors.append("edge publisher no longer retains telemetry for application ACK")
    if "ack_topic(record)" not in cloud_ingest or "ack_payload(record, ACK_HMAC_KEY)" not in cloud_ingest:
        errors.append("ingestor post-persistence application ACK contract is missing")
    if "expected_ack_key" not in smoke or "ack_matches_expected" not in smoke:
        errors.append("integration smoke no longer verifies the exact application ACK identity")
    ack_edge = (ROOT / "edge/railguard_edge/ack.py").read_text()
    ack_cloud = (ROOT / "cloud/ingestor/ack.py").read_text()
    if "hmac.compare_digest" not in ack_edge or "hmac_sha256" not in ack_cloud:
        errors.append("post-commit ACK authentication contract is missing")
    if "ACK_HMAC_KEY=" not in (ROOT / ".env.example").read_text():
        errors.append("reference stack no longer configures ACK HMAC authentication")

    init_sql = (ROOT / "cloud/db/init.sql").read_text()
    processor = (ROOT / "cloud/ingestor/processor.py").read_text()
    if "PRIMARY KEY (device_id, ts, seq)" not in init_sql or "ON CONFLICT (device_id, ts, seq)" not in processor:
        errors.append("database idempotency identity no longer matches application ACK identity")
    if "source_seq BIGINT NOT NULL" not in init_sql or "source_seq" not in processor:
        errors.append("prediction provenance no longer preserves source telemetry sequence")
    ts_contract = schema.get("properties", {}).get("ts", {})
    if not ts_contract.get("pattern", "").endswith("Z$"):
        errors.append("telemetry timestamp no longer requires canonical UTC Z identity form")

    ci = (ROOT / ".github/workflows/ci.yml").read_text()
    if "integration-stack:" not in ci or "scripts/integration_smoke.py" not in ci:
        errors.append("composed MQTT/TimescaleDB/API integration smoke CI job is missing")
    return errors


def main() -> int:
    ap = argparse.ArgumentParser(description="Synchronize/check measured benchmark evidence in portfolio docs.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="fail if checked-in evidence differs from benchmark JSON")
    mode.add_argument("--write", action="store_true", help="rewrite benchmark evidence blocks")
    args = ap.parse_args()
    data = json.loads(RESULTS.read_text())
    expected = expected_files(data)
    if args.check:
        drift = [str(path.relative_to(ROOT)) for path, text in expected.items() if path.read_text() != text]
        errors = contract_checks(data)
        if drift or errors:
            for path in drift:
                print(f"evidence drift: {path}", file=sys.stderr)
            for error in errors:
                print(f"evidence check: {error}", file=sys.stderr)
            return 1
        print("evidence checks: PASS")
        return 0
    for path, text in expected.items():
        path.write_text(text)
    print("benchmark evidence synchronized")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
