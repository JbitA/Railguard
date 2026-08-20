from __future__ import annotations

import argparse
import json
import sys



def _reject_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def parse_native_record(line: str) -> dict:
    record = json.loads(line, parse_constant=_reject_constant)
    if not isinstance(record, dict):
        raise ValueError("native telemetry line must decode to a JSON object")
    return record


def main():
    from .config import load_config
    from .publisher import TelemetryPublisher

    p = argparse.ArgumentParser(description="Bridge native C++ NDJSON telemetry into the durable MQTT publisher")
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = load_config(args.config)
    pub = TelemetryPublisher(cfg.mqtt, cfg.spool_path, max_records=cfg.spool_max_records)
    pub.start()
    rejected = 0
    try:
        for line in sys.stdin:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = parse_native_record(line)
                if record.get("device_id") != cfg.device_id:
                    raise ValueError(
                        f"native record device_id={record.get('device_id')!r} does not match configured device_id={cfg.device_id!r}"
                    )
                pub.publish(record)
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                rejected += 1
                print(f"native bridge rejected line #{rejected}: {exc}", file=sys.stderr, flush=True)
    finally:
        pub.close()


if __name__ == "__main__":
    main()
