from __future__ import annotations
import json


def flatten(d: dict) -> dict:
    bands = list(d["vibration"].get("band_energy", [0, 0, 0, 0])) + [0, 0, 0, 0]
    sensors = d["vibration"].get("sensors", [])

    def sensor_rms(sensor_id: int):
        return next((x.get("rms_ms2") for x in sensors if x.get("sensor_id") == sensor_id), None)

    return {
        "schema_version": d["schema_version"],
        "ts": d["ts"], "device_id": d["device_id"], "seq": d["seq"],
        "sample_period_ms": d["sample_period_ms"],
        "lat": d["gps"].get("lat"), "lon": d["gps"].get("lon"), "speed": d["gps"].get("speed_mps"),
        "temp": d["environment"].get("temperature_c"), "humidity": d["environment"].get("humidity"),
        "rms": d["vibration"].get("rms_ms2"), "peak": d["vibration"].get("peak_ms2"),
        "kurt": d["vibration"].get("kurtosis"), "crest": d["vibration"].get("crest_factor"),
        "b0": bands[0], "b1": bands[1], "b2": bands[2], "b3": bands[3],
        "motion": d["vision"].get("motion_score"), "contrast": d["vision"].get("contrast"), "sharpness": d["vision"].get("sharpness"),
        "loss": d["health"].get("packet_loss", 0), "spool": d["health"].get("spool_depth", 0), "spool_dropped": d["health"].get("spool_dropped", 0),
        "camera_matched": d["health"].get("camera_matched"), "sync_error_ms": d["health"].get("sync_error_ms"),
        "sensor_skew_ms": d["health"].get("sensor_skew_ms"),
        "clock_alignment_locked": d["health"].get("clock_alignment_locked"),
        "clock_jitter_ms": d["health"].get("clock_jitter_ms"),
        "clock_samples": d["health"].get("clock_samples"),
        "context_flags": d["health"].get("context_flags", 0),
        "sensor0_rms": sensor_rms(0), "sensor1_rms": sensor_rms(1), "sensor2_rms": sensor_rms(2),
        "raw": json.dumps(d),
    }
