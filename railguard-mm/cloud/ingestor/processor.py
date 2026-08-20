from __future__ import annotations

try:
    from .transform import flatten
    from .validation import validate_record
except ImportError:  # direct execution inside the ingestor container
    from transform import flatten
    from validation import validate_record

INSERT = """
INSERT INTO telemetry (
  ts, device_id, seq, schema_version, sample_period_ms, lat, lon, speed_mps, temperature_c, humidity,
  vibration_rms, vibration_peak, vibration_kurtosis, crest_factor,
  band0, band1, band2, band3, vision_motion, vision_contrast, vision_sharpness,
  packet_loss, spool_depth, spool_dropped, camera_matched, sync_error_ms, sensor_skew_ms,
  clock_alignment_locked, clock_jitter_ms, clock_samples, context_flags,
  sensor0_rms, sensor1_rms, sensor2_rms, raw
) VALUES (
  %(ts)s, %(device_id)s, %(seq)s, %(schema_version)s, %(sample_period_ms)s, %(lat)s, %(lon)s, %(speed)s, %(temp)s, %(humidity)s,
  %(rms)s, %(peak)s, %(kurt)s, %(crest)s,
  %(b0)s, %(b1)s, %(b2)s, %(b3)s, %(motion)s, %(contrast)s, %(sharpness)s,
  %(loss)s, %(spool)s, %(spool_dropped)s, %(camera_matched)s, %(sync_error_ms)s, %(sensor_skew_ms)s,
  %(clock_alignment_locked)s, %(clock_jitter_ms)s, %(clock_samples)s, %(context_flags)s,
  %(sensor0_rms)s, %(sensor1_rms)s, %(sensor2_rms)s, %(raw)s::jsonb
)
ON CONFLICT (device_id, ts, seq) DO UPDATE SET
  seq = EXCLUDED.seq,
  schema_version = EXCLUDED.schema_version,
  sample_period_ms = EXCLUDED.sample_period_ms,
  lat = EXCLUDED.lat, lon = EXCLUDED.lon, speed_mps = EXCLUDED.speed_mps,
  temperature_c = EXCLUDED.temperature_c, humidity = EXCLUDED.humidity,
  vibration_rms = EXCLUDED.vibration_rms, vibration_peak = EXCLUDED.vibration_peak,
  vibration_kurtosis = EXCLUDED.vibration_kurtosis, crest_factor = EXCLUDED.crest_factor,
  band0 = EXCLUDED.band0, band1 = EXCLUDED.band1, band2 = EXCLUDED.band2, band3 = EXCLUDED.band3,
  vision_motion = EXCLUDED.vision_motion, vision_contrast = EXCLUDED.vision_contrast, vision_sharpness = EXCLUDED.vision_sharpness,
  packet_loss = EXCLUDED.packet_loss, spool_depth = EXCLUDED.spool_depth, spool_dropped = EXCLUDED.spool_dropped,
  camera_matched = EXCLUDED.camera_matched, sync_error_ms = EXCLUDED.sync_error_ms,
  sensor_skew_ms = EXCLUDED.sensor_skew_ms,
  clock_alignment_locked = EXCLUDED.clock_alignment_locked,
  clock_jitter_ms = EXCLUDED.clock_jitter_ms, clock_samples = EXCLUDED.clock_samples,
  context_flags = EXCLUDED.context_flags,
  sensor0_rms = EXCLUDED.sensor0_rms, sensor1_rms = EXCLUDED.sensor1_rms, sensor2_rms = EXCLUDED.sensor2_rms,
  raw = EXCLUDED.raw;
"""

PREDICTION_INSERT = """
INSERT INTO predictions(issued_at,target_ts,device_id,source_seq,model_version,horizon_steps,step_ms,vibration_rms_pred,vision_motion_pred,anomaly_probability)
VALUES (%s, %s::timestamptz + (%s * interval '1 millisecond'), %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (device_id, issued_at, source_seq, target_ts, model_version) DO UPDATE SET
  horizon_steps=EXCLUDED.horizon_steps, step_ms=EXCLUDED.step_ms,
  vibration_rms_pred=EXCLUDED.vibration_rms_pred,
  vision_motion_pred=EXCLUDED.vision_motion_pred,
  anomaly_probability=EXCLUDED.anomaly_probability
"""


def persist_prediction(conn, record: dict) -> int:
    pred = record.get("prediction")
    if not pred:
        return 0
    horizons = pred["horizons"]
    vib = pred["vibration_rms"]
    vis = pred["vision_motion"]
    step_ms = float(pred["step_ms"])
    for i, horizon in enumerate(horizons):
        conn.execute(
            PREDICTION_INSERT,
            (
                record["ts"], record["ts"], float(horizon) * step_ms, record["device_id"], int(record["seq"]), pred["model_version"],
                int(horizon), step_ms, float(vib[i]), float(vis[i]), float(pred["anomaly_probability"]),
            ),
        )
    return len(horizons)


def process_record(conn, validator, record: dict) -> tuple[dict, int]:
    """Validate and persist one telemetry record using an injected DB connection.

    The pure dependency boundary makes schema rejection, SQL mapping and prediction
    fan-out testable without starting MQTT or requiring a live TimescaleDB instance.
    """
    validate_record(validator, record)
    row = flatten(record)
    conn.execute(INSERT, row)
    prediction_rows = persist_prediction(conn, record)
    return row, prediction_rows


def topic_matches_device(topic: str, device_id: str) -> bool:
    prefix = "railguard/telemetry/"
    return topic.startswith(prefix) and topic[len(prefix):] == device_id and "/" not in device_id
