from __future__ import annotations

import numpy as np
import pandas as pd


def validate_modeling_table(df: pd.DataFrame, sensor_columns: list[str]) -> None:
    """Fail closed on malformed/non-finite supervised modeling inputs.

    The deployed native runtime refuses inference when required context is invalid;
    training must not silently use a weaker contract by allowing NaN/Inf values to
    flow through normalization.  Anomaly labels are intentionally excluded because
    unlabeled samples are represented by NaN and gated separately.
    """
    required = list(dict.fromkeys(list(sensor_columns) + ["vibration_rms", "vision_motion", "image_path", "run_id", "ts"]))
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"modeling table missing required columns: {missing}")
    if df.empty:
        raise ValueError("modeling table is empty")

    numeric = list(dict.fromkeys(list(sensor_columns) + ["vibration_rms", "vision_motion"]))
    bad: dict[str, int] = {}
    for name in numeric:
        values = pd.to_numeric(df[name], errors="coerce").to_numpy(float)
        count = int((~np.isfinite(values)).sum())
        if count:
            bad[name] = count
    if bad:
        detail = ", ".join(f"{name}={count}" for name, count in bad.items())
        raise ValueError(f"modeling table contains non-finite required values: {detail}")

    for name in ("vibration_rms", "vibration_peak", "vibration_kurtosis", "crest_factor", "speed_mps"):
        if name in df.columns and (pd.to_numeric(df[name], errors="coerce") < 0).any():
            raise ValueError(f"modeling table contains negative physical values in {name}")
    if "humidity" in df.columns:
        humidity = pd.to_numeric(df["humidity"], errors="coerce")
        if ((humidity < 0) | (humidity > 1)).any():
            raise ValueError("modeling table humidity must be in [0,1]")
    motion = pd.to_numeric(df["vision_motion"], errors="coerce")
    if ((motion < 0) | (motion > 1)).any():
        raise ValueError("modeling table vision_motion must be in [0,1]")
    if (df["image_path"].astype(str).str.len() == 0).any():
        raise ValueError("modeling table contains an empty image_path")
