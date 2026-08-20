from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.stats import kurtosis

CHANNELS = [f"Channel_{i}" for i in range(1, 7)]


def image_ts(path: Path) -> float | None:
    nums = re.findall(r"\d{10}(?:\.\d+)?|\d{13}", path.stem)
    if not nums:
        return None
    x = float(nums[-1])
    return x / 1000.0 if x > 1e11 else x


def nearest_image(ts: float, image_times: np.ndarray, paths: list[Path]) -> tuple[str, float]:
    i = int(np.searchsorted(image_times, ts))
    cand: list[int] = []
    if i < len(paths):
        cand.append(i)
    if i > 0:
        cand.append(i - 1)
    if not cand:
        return "", float("inf")
    j = min(cand, key=lambda k: abs(image_times[k] - ts))
    return str(paths[j]), float(abs(image_times[j] - ts))


def haversine(lat1, lon1, lat2, lon2):
    r = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_anomaly_catalog(path: Path | None, default_radius_m: float) -> list[dict]:
    if path is None:
        return []
    df = pd.read_csv(path)
    required = {"latitude", "longitude"}
    if not required.issubset(df.columns):
        raise ValueError(f"anomaly catalog must contain {sorted(required)}")
    out = []
    for i, row in df.iterrows():
        out.append({
            "id": str(row.get("anomaly_id", f"anomaly-{i+1}")),
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "radius_m": float(row.get("radius_m", default_radius_m)),
        })
    return out


def anomaly_label(lat: float, lon: float, catalog: list[dict]) -> tuple[float, str | None, float | None]:
    """Return supervised label only when a ground-truth catalog is supplied.

    Without a catalog the correct value is unknown (NaN), not zero.  This prevents
    the anomaly head from being silently trained as if the complete track were normal.
    """
    if not catalog or not np.isfinite(lat) or not np.isfinite(lon):
        return float("nan"), None, None
    distances = [(haversine(lat, lon, a["latitude"], a["longitude"]), a) for a in catalog]
    distance, closest = min(distances, key=lambda x: x[0])
    label = 1.0 if distance <= closest["radius_m"] else 0.0
    return label, closest["id"] if label else None, float(distance)


def _default_run_id(csv_path: Path) -> str:
    parent = csv_path.parent
    if parent.parent != parent:
        return f"{parent.parent.name}/{parent.name}"
    return parent.name


def alignment_quality_summary(errors_ms: list[float], *, total_bins: int, rejected_far_image: int) -> dict:
    values = np.asarray(errors_ms, dtype=float)
    finite = values[np.isfinite(values)]
    return {
        "total_time_bins": int(total_bins),
        "accepted_windows": int(finite.size),
        "rejected_far_image": int(rejected_far_image),
        "acceptance_rate": float(finite.size / total_bins) if total_bins else 0.0,
        "image_time_error_ms": {
            "p50": float(np.percentile(finite, 50)) if finite.size else None,
            "p95": float(np.percentile(finite, 95)) if finite.size else None,
            "max": float(np.max(finite)) if finite.size else None,
        },
    }


def prepare(
    csv_path: Path,
    out: Path,
    window_s: float = 0.1,
    source_timezone: str = "America/New_York",
    run_id: str | None = None,
    anomaly_catalog: Path | None = None,
    anomaly_radius_m: float = 15.0,
    max_image_delta_s: float = 0.25,
    qa_out: Path | None = None,
):
    df = pd.read_csv(csv_path)
    missing = [c for c in CHANNELS if c not in df.columns]
    if missing:
        raise SystemExit(f"missing expected channels: {missing}")

    # Dataset CSV timestamps are local wall-clock strings; image filenames use Unix time.
    ts = pd.to_datetime(df["Timestamp"], errors="coerce")
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(source_timezone, ambiguous="infer", nonexistent="shift_forward").dt.tz_convert("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    valid = ts.notna()
    df = df.loc[valid].copy()
    ts = ts.loc[valid]
    df["unix"] = ts.astype("int64") / 1e9

    images = sorted(list(csv_path.parent.rglob("*.jpg")) + list(csv_path.parent.rglob("*.JPG")))
    pairs = sorted((t, p) for p in images if (t := image_ts(p)) is not None)
    if not pairs:
        raise SystemExit("no image filenames with Unix-like timestamps found")
    image_times = np.array([x[0] for x in pairs])
    image_paths = [x[1] for x in pairs]

    catalog = load_anomaly_catalog(anomaly_catalog, anomaly_radius_m)
    rid = run_id or _default_run_id(csv_path)
    t0 = float(df.unix.iloc[0])
    df["bin"] = np.floor((df.unix - t0) / window_s).astype(int)

    rows = []
    image_errors_ms: list[float] = []
    rejected_far_image = 0
    total_bins = int(df["bin"].nunique())
    prev_img = None
    prev_img_path = None
    prev_lat = prev_lon = prev_t = None

    for _, g in df.groupby("bin", sort=True):
        target = float(g.unix.mean())
        vals = g[CHANNELS].to_numpy(float)
        channel_rms = np.sqrt(np.mean(vals * vals, axis=0))
        rms = float(np.sqrt(np.mean(channel_rms * channel_rms)))
        peak = float(np.max(np.abs(vals)))
        channel_kurtosis = [float(kurtosis(vals[:, i], fisher=False, bias=False)) if len(vals) > 8 else 3.0 for i in range(vals.shape[1])]
        k = float(np.nanmean(channel_kurtosis))

        image_path, image_delta_s = nearest_image(target, image_times, image_paths)
        if image_delta_s > max_image_delta_s:
            # A distant frame is worse than an explicit missing observation.  Dropping
            # this modeling row preserves the fixed-rate sequence contract.
            rejected_far_image += 1
            continue
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (320, 180))
        contrast = float(np.std(img) / 64.0)
        sharpness = float(cv2.Laplacian(img, cv2.CV_64F).var() / 1000.0)
        motion = 0.0
        if prev_img is not None and prev_img_path != image_path:
            motion = float(np.mean(cv2.absdiff(prev_img, img)) / 255.0)
        prev_img, prev_img_path = img, image_path

        lat = float(g["Latitude"].mean()) if "Latitude" in g else float("nan")
        lon = float(g["Longitude"].mean()) if "Longitude" in g else float("nan")
        speed = 0.0
        if prev_t is not None and target > prev_t and all(np.isfinite(x) for x in (prev_lat, prev_lon, lat, lon)):
            speed = haversine(prev_lat, prev_lon, lat, lon) / (target - prev_t)
        prev_lat, prev_lon, prev_t = lat, lon, target

        temp_f = float(g["Temperature"].mean()) if "Temperature" in g else float("nan")
        temp = (temp_f - 32.0) * (5.0 / 9.0) if np.isfinite(temp_f) else float("nan")
        hum = float(g["Humidity"].mean()) if "Humidity" in g else float("nan")
        anomaly, anomaly_id, anomaly_distance_m = anomaly_label(lat, lon, catalog)

        row = {
            "run_id": rid,
            "ts": pd.to_datetime(target, unit="s", utc=True).isoformat(),
            "image_path": image_path,
            "image_time_error_ms": 1000.0 * image_delta_s,
            "vibration_rms": rms,
            "vibration_peak": peak,
            "vibration_kurtosis": k,
            "crest_factor": peak / (rms + 1e-9),
            "vision_motion": motion,
            "vision_contrast": contrast,
            "vision_sharpness": sharpness,
            "speed_mps": speed,
            "temperature_c": temp,
            "humidity": hum,
            "latitude": lat,
            "longitude": lon,
            "anomaly": anomaly,
            "anomaly_id": anomaly_id,
            "anomaly_distance_m": anomaly_distance_m,
        }
        for i, value in enumerate(channel_rms, start=1):
            row[f"accel_{i}_rms"] = float(value)
        rows.append(row)
        image_errors_ms.append(1000.0 * image_delta_s)

    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    qa = alignment_quality_summary(image_errors_ms, total_bins=total_bins, rejected_far_image=rejected_far_image)
    qa.update({"run_id": rid, "max_image_delta_ms": 1000.0 * max_image_delta_s, "anomaly_supervision": "catalog" if catalog else "unlabeled"})
    qa_path = qa_out or out.with_suffix(".qa.json")
    qa_path.write_text(json.dumps(qa, indent=2) + "\n")
    supervision = "catalog" if catalog else "unlabeled"
    print(f"wrote {len(rows)} windows to {out} run_id={rid} anomaly_supervision={supervision} qa={qa_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("csv", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--window", type=float, default=0.1)
    p.add_argument("--timezone", default="America/New_York", help="timezone of naive CSV Timestamp values")
    p.add_argument("--run-id", default=None, help="stable physical-run identifier; default uses the last two path components")
    p.add_argument("--anomaly-catalog", type=Path, default=None, help="CSV with latitude,longitude[,radius_m,anomaly_id]")
    p.add_argument("--anomaly-radius-m", type=float, default=15.0, help="default catalog radius when radius_m is absent")
    p.add_argument("--max-image-delta", type=float, default=0.25, help="reject nearest image matches farther than this many seconds")
    p.add_argument("--qa-out", type=Path, default=None, help="alignment QA JSON; defaults beside the processed CSV")
    a = p.parse_args()
    prepare(a.csv, a.out, a.window, a.timezone, a.run_id, a.anomaly_catalog, a.anomaly_radius_m, a.max_image_delta, a.qa_out)


if __name__ == "__main__":
    main()
