from __future__ import annotations
import math
import os
import time
from datetime import timedelta
import numpy as np

DB = os.getenv("DATABASE_URL", "postgresql://railguard:railguard_dev@timescaledb:5432/railguard")
MODEL_VERSION = os.getenv("MODEL_VERSION", "cloud-trend-baseline-v2")
HORIZONS = (1, 5, 10)


def _trend_forecast(
    values: np.ndarray,
    horizons: tuple[int, ...],
    *,
    lower: float | None = None,
    upper: float | None = None,
) -> list[float]:
    n = min(16, len(values))
    y = values[-n:]
    x = np.arange(n, dtype=float)
    if n < 2 or float(np.std(y)) < 1e-12:
        raw = [float(y[-1])] * len(horizons)
    else:
        slope, intercept = np.polyfit(x, y, 1)
        # Blend the trend with the last value so a noisy short window cannot extrapolate wildly.
        raw = [float(0.65 * y[-1] + 0.35 * (intercept + slope * (n - 1 + h))) for h in horizons]
    if lower is not None:
        raw = [max(lower, value) for value in raw]
    if upper is not None:
        raw = [min(upper, value) for value in raw]
    return raw


def infer_one(conn, device_id: str):
    rows = conn.execute("""
      SELECT ts, seq, sample_period_ms, vibration_rms, vision_motion, speed_mps
      FROM telemetry WHERE device_id=%s ORDER BY ts DESC, seq DESC LIMIT 32
    """, (device_id,)).fetchall()
    if len(rows) < 8:
        return
    rows = list(reversed(rows))
    vib = np.array([r["vibration_rms"] or 0.0 for r in rows], dtype=float)
    vis = np.array([r["vision_motion"] or 0.0 for r in rows], dtype=float)
    step_ms = float(rows[-1]["sample_period_ms"] or 100.0)
    vib_pred = _trend_forecast(vib, HORIZONS, lower=0.0)
    vis_pred = _trend_forecast(vis, HORIZONS, lower=0.0, upper=1.0)

    vib_sigma = np.std(vib[-16:]) + 1e-6
    residual_z = abs(vib[-1] - np.mean(vib[-8:-1])) / vib_sigma
    anomaly = 1.0 / (1.0 + math.exp(-(residual_z - 2.0)))
    issued = rows[-1]["ts"]
    source_seq = int(rows[-1]["seq"])

    for h, vp, mp in zip(HORIZONS, vib_pred, vis_pred):
        target = issued + timedelta(milliseconds=step_ms * h)
        conn.execute("""
          INSERT INTO predictions(issued_at,target_ts,device_id,source_seq,model_version,horizon_steps,step_ms,
                                  vibration_rms_pred,vision_motion_pred,anomaly_probability)
          VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
          ON CONFLICT (device_id, issued_at, source_seq, target_ts, model_version) DO UPDATE SET
            horizon_steps=EXCLUDED.horizon_steps, step_ms=EXCLUDED.step_ms,
            vibration_rms_pred=EXCLUDED.vibration_rms_pred,
            vision_motion_pred=EXCLUDED.vision_motion_pred,
            anomaly_probability=EXCLUDED.anomaly_probability
        """, (issued, target, device_id, source_seq, MODEL_VERSION, h, step_ms, vp, mp, float(anomaly)))


def run_forever(*, poll_interval_s: float = 1.0) -> None:
    """Run the optional cloud trend baseline.

    Kept behind an explicit entry point so importing this module in tests or tooling
    never opens a database connection or starts an infinite loop.
    """
    import psycopg
    from psycopg.rows import dict_row

    while True:
        try:
            with psycopg.connect(DB, row_factory=dict_row, autocommit=True) as conn:
                devices = conn.execute("SELECT DISTINCT device_id FROM telemetry").fetchall()
                for row in devices:
                    infer_one(conn, row["device_id"])
        except Exception as e:
            print("worker error:", e, flush=True)
        time.sleep(poll_interval_s)


if __name__ == "__main__":
    run_forever()
