from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

try:
    from railguard_ml.dataset import RailSequenceDataset
    from railguard_ml.models import FusionTransformer
    from railguard_ml.splits import select_runs, select_spatial_blocks
    from railguard_ml.provenance import verify_dataset_fingerprint
    from railguard_ml.contracts import DEPLOYMENT_IMAGE_MODE, MODEL_ARCH_VERSION
    from railguard_ml.validation import validate_modeling_table
    from railguard_ml.modalities import VISUAL_STRUCTURED_COLUMNS, forward_for_modality
except ModuleNotFoundError:  # package import during repository tests
    from ml.railguard_ml.dataset import RailSequenceDataset
    from ml.railguard_ml.models import FusionTransformer
    from ml.railguard_ml.splits import select_runs, select_spatial_blocks
    from ml.railguard_ml.provenance import verify_dataset_fingerprint
    from ml.railguard_ml.contracts import DEPLOYMENT_IMAGE_MODE, MODEL_ARCH_VERSION
    from ml.railguard_ml.validation import validate_modeling_table
    from ml.railguard_ml.modalities import VISUAL_STRUCTURED_COLUMNS, forward_for_modality

HORIZONS = (1, 5, 10)


def _masked_sensors(sensors: torch.Tensor, columns: list[str], *, keep_visual: bool) -> torch.Tensor:
    """Mask normalized structured features at their training-mean value (zero z-score)."""
    masked = sensors.clone()
    for idx, name in enumerate(columns):
        is_visual = name in VISUAL_STRUCTURED_COLUMNS
        if is_visual != keep_visual:
            masked[..., idx] = 0.0
    return masked


def _mae(pred: np.ndarray, target: np.ndarray):
    return np.mean(np.abs(pred - target), axis=0).tolist()


def _rmse(pred: np.ndarray, target: np.ndarray):
    return np.sqrt(np.mean((pred - target) ** 2, axis=0)).tolist()


def shift_frames(frames: torch.Tensor, offset_steps: int) -> torch.Tensor:
    """Shift camera observations relative to sensors without wrapping future frames.

    Positive offsets mean the camera stream lags the sensor stream. Edge values are
    repeated instead of circularly wrapped so the ablation cannot leak future images.
    """
    if offset_steps == 0:
        return frames
    t = frames.shape[1]
    if abs(offset_steps) >= t:
        raise ValueError("frame offset must be smaller than sequence length")
    out = torch.empty_like(frames)
    if offset_steps > 0:
        out[:, :offset_steps] = frames[:, :1].expand(-1, offset_steps, -1, -1, -1)
        out[:, offset_steps:] = frames[:, :-offset_steps]
    else:
        k = -offset_steps
        out[:, :-k] = frames[:, k:]
        out[:, -k:] = frames[:, -1:].expand(-1, k, -1, -1, -1)
    return out


def hold_last_frame_dropout(frames: torch.Tensor, probability: float, generator: torch.Generator) -> torch.Tensor:
    """Simulate dropped camera frames using a hold-last-sample policy."""
    if probability <= 0.0:
        return frames
    if probability >= 1.0:
        probability = 1.0
    out = frames.clone()
    mask = torch.rand((frames.shape[0], frames.shape[1]), generator=generator) < probability
    # Preserve the first frame as an anchor, then hold the last available observation.
    mask[:, 0] = False
    for step in range(1, frames.shape[1]):
        replace = mask[:, step].to(frames.device)
        if bool(replace.any()):
            out[replace, step] = out[replace, step - 1]
    return out


def _collect(model, dl, device, mode: str, sensor_columns: list[str], *, frame_offset_steps: int = 0, frame_dropout: float = 0.0, seed: int = 7):
    pv, pvis, tv, tvis, anom, labels, label_valid, curv, curvis = [], [], [], [], [], [], [], [], []
    model.eval()
    dropout_generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for b in dl:
            frames = b["frames"].to(device)
            if frame_offset_steps:
                frames = shift_frames(frames, frame_offset_steps)
            if frame_dropout > 0.0:
                frames = hold_last_frame_dropout(frames, frame_dropout, dropout_generator)
            sensors = b["sensors"].to(device)
            if mode in ("multimodal", "sensor_only", "vision_only"):
                out = forward_for_modality(model, frames, sensors, sensor_columns, mode)
            elif mode == "nonvisual_inputs_ablation":
                # No camera embedding and no engineered visual scalars. Remaining structured
                # inputs are vibration + speed/environment context.
                out = model.forward_ablated(
                    frames, _masked_sensors(sensors, sensor_columns, keep_visual=False),
                    use_vision=False, use_sensors=True)
            elif mode == "visual_inputs_ablation":
                # Keep image embedding + engineered visual scalars; set vibration/context
                # z-scores to zero (their training means).
                out = model.forward_ablated(
                    frames, _masked_sensors(sensors, sensor_columns, keep_visual=True),
                    use_vision=True, use_sensors=True)
            else:
                raise ValueError(mode)
            pv.append(out["vibration"].cpu().numpy())
            pvis.append(out["vision"].cpu().numpy())
            tv.append(b["vibration_target"].numpy())
            tvis.append(b["vision_target"].numpy())
            anom.append(torch.sigmoid(out["anomaly_logit"]).cpu().numpy())
            labels.append(b["anomaly"].numpy())
            label_valid.append(b["anomaly_valid"].numpy())
            curv.append(b["current_vibration"].numpy())
            curvis.append(b["current_vision"].numpy())
    return {
        "pv": np.concatenate(pv),
        "pvis": np.concatenate(pvis),
        "tv": np.concatenate(tv),
        "tvis": np.concatenate(tvis),
        "anom": np.concatenate(anom),
        "labels": np.concatenate(labels),
        "label_valid": np.concatenate(label_valid).astype(bool),
        "curv": np.concatenate(curv),
        "curvis": np.concatenate(curvis),
    }


def _metrics(x: dict) -> dict:
    result = {
        "vibration_mae": _mae(x["pv"], x["tv"]),
        "vibration_rmse": _rmse(x["pv"], x["tv"]),
        "vision_mae": _mae(x["pvis"], x["tvis"]),
        "vision_rmse": _rmse(x["pvis"], x["tvis"]),
        "vibration_mae_mean": float(np.mean(np.abs(x["pv"] - x["tv"]))),
        "vision_mae_mean": float(np.mean(np.abs(x["pvis"] - x["tvis"]))),
    }
    valid = x["label_valid"]
    if valid.any() and np.unique(x["labels"][valid]).size > 1:
        result["anomaly_auroc"] = float(roc_auc_score(x["labels"][valid], x["anom"][valid]))
        result["anomaly_average_precision"] = float(average_precision_score(x["labels"][valid], x["anom"][valid]))
        result["anomaly_labeled_windows"] = int(valid.sum())
    return result


def _persistence(x: dict) -> dict:
    pv = np.repeat(x["curv"][:, None], 3, axis=1)
    pvis = np.repeat(x["curvis"][:, None], 3, axis=1)
    y = dict(x)
    y["pv"], y["pvis"] = pv, pvis
    return _metrics(y)


def summarize_group_variability(per_group: dict[str, dict]) -> dict:
    """Summarize between-group generalization instead of hiding it in one global MAE."""
    if not per_group:
        return {"groups_with_windows": 0}

    mm = np.asarray([v["multimodal"]["vibration_mae_mean"] for v in per_group.values()], dtype=float)
    persistence = np.asarray([v["persistence"]["vibration_mae_mean"] for v in per_group.values()], dtype=float)
    improvement = np.where(persistence > 1e-12, (persistence - mm) / persistence, np.nan)

    def stats(values: np.ndarray) -> dict:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return {"n": 0}
        return {
            "n": int(finite.size),
            "median": float(np.median(finite)),
            "p25": float(np.quantile(finite, 0.25)),
            "p75": float(np.quantile(finite, 0.75)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }

    return {
        "groups_with_windows": len(per_group),
        "multimodal_vibration_mae_mean": stats(mm),
        "persistence_vibration_mae_mean": stats(persistence),
        "relative_vibration_mae_improvement": stats(improvement),
    }


def markdown(report: dict) -> str:
    lines = [
        "# Multimodal untouched-test evaluation",
        "",
        f"Split mode: **{report['split_mode']}**  ",
        f"Untouched test groups: **{report['test_group_count']}** — `{', '.join(report.get('test_groups', [])[:8])}`{' …' if report['test_group_count'] > 8 else ''}  ",
        f"Checkpoint-selection groups: **{report['validation_group_count']}** — `{', '.join(report.get('validation_groups', [])[:8])}`{' …' if report['validation_group_count'] > 8 else ''}  ",
        f"Windows: **{report['windows']}**  ",
        f"Model step: **{report['sample_period_ms']:.1f} ms**",
        "",
        "| Configuration | Vib MAE +1 | +5 | +10 | Vision MAE +1 | +5 | +10 | Anomaly AUROC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["persistence", "nonvisual_inputs_ablation", "visual_inputs_ablation", "multimodal"]:
        m = report["metrics"][name]
        v, w, auc = m["vibration_mae"], m["vision_mae"], m.get("anomaly_auroc")
        auc_text = f"{auc:.4f}" if auc is not None else "n/a"
        lines.append(
            f"| {name} | {v[0]:.5f} | {v[1]:.5f} | {v[2]:.5f} | "
            f"{w[0]:.5f} | {w[1]:.5f} | {w[2]:.5f} | {auc_text} |"
        )
    variability = report.get("group_variability", {})
    per_group = variability.get("per_group", {})
    if per_group:
        lines += ["", "## Held-out group variability", "",
                  "| Test group | Windows | Persistence vib MAE | Multimodal vib MAE | Relative improvement |",
                  "|---|---:|---:|---:|---:|"]
        for group, item in per_group.items():
            p_mae = item["persistence"]["vibration_mae_mean"]
            m_mae = item["multimodal"]["vibration_mae_mean"]
            rel = (p_mae - m_mae) / p_mae if p_mae > 1e-12 else float("nan")
            rel_text = f"{rel:.1%}" if np.isfinite(rel) else "n/a"
            lines.append(f"| {group} | {item['windows']} | {p_mae:.5f} | {m_mae:.5f} | {rel_text} |")
        summary = variability.get("summary", {})
        mm_summary = summary.get("multimodal_vibration_mae_mean", {})
        if mm_summary.get("n"):
            lines += ["",
                      f"Across {mm_summary['n']} test groups, multimodal vibration MAE median="
                      f"**{mm_summary['median']:.5f}** with IQR "
                      f"[{mm_summary['p25']:.5f}, {mm_summary['p75']:.5f}]."]

    lines += ["", "## Camera synchronization robustness", "",
              "| Camera offset (steps) | Vib MAE mean | Vision MAE mean |",
              "|---:|---:|---:|"]
    for offset, m in report.get("sensitivity", {}).get("camera_offset_steps", {}).items():
        lines.append(f"| {offset} | {m['vibration_mae_mean']:.5f} | {m['vision_mae_mean']:.5f} |")
    lines += ["", "| Hold-last frame dropout | Vib MAE mean | Vision MAE mean |",
              "|---:|---:|---:|"]
    for probability, m in report.get("sensitivity", {}).get("frame_dropout", {}).items():
        lines.append(f"| {float(probability):.0%} | {m['vibration_mae_mean']:.5f} | {m['vision_mae_mean']:.5f} |")
    lines += [
        "",
        "Camera-offset sensitivity deliberately shifts image sequences without circular wrap; frame-dropout sensitivity repeats the last available frame. "
        "Both are inference-time robustness tests on the untouched test split, not additional training augmentations.",
        "",
        "The ablation rows mask normalized structured features at their training-mean z-score and/or remove the camera embedding. "
        "`nonvisual_inputs_ablation` retains vibration + speed/environment context only; `visual_inputs_ablation` retains camera frames + engineered visual scalars only. "
        "They measure dependency of one already-trained fusion model, not the performance of separately trained unimodal models.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Untouched-test multimodal evaluation with dependency ablations.")
    ap.add_argument("table", type=Path)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out-json", type=Path, default=Path("artifacts/evaluation/multimodal_metrics.json"))
    ap.add_argument("--out-md", type=Path, default=Path("artifacts/evaluation/multimodal_metrics.md"))
    ap.add_argument("--timing-offset-steps", default="1,2", help="comma-separated fixed camera offsets for sensitivity analysis")
    ap.add_argument("--frame-dropout", default="0.1,0.3", help="comma-separated camera dropout probabilities using hold-last-frame")
    ap.add_argument("--sensitivity-seed", type=int, default=7)
    a = ap.parse_args()

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    if ck.get("training_modality", "multimodal") != "multimodal":
        raise SystemExit("dependency-ablation evaluation requires a multimodal checkpoint; use evaluate_independent_models.py for separately trained baselines")
    if int(ck.get("model_arch_version", -1)) != MODEL_ARCH_VERSION:
        raise SystemExit(
            f"checkpoint model_arch_version={ck.get('model_arch_version')} is incompatible with evaluator version {MODEL_ARCH_VERSION}; retrain/export explicitly"
        )
    if ck.get("image_mode") != DEPLOYMENT_IMAGE_MODE:
        raise SystemExit(
            f"checkpoint image_mode={ck.get('image_mode')!r} does not match deployment image contract {DEPLOYMENT_IMAGE_MODE!r}; retrain explicitly"
        )
    expected_fingerprint = (ck.get("provenance") or {}).get("dataset_fingerprint")
    if not expected_fingerprint:
        raise SystemExit(
            "checkpoint predates multimodal dataset fingerprinting. Retrain with the current training pipeline; "
            "the untouched-test evaluator will not score a checkpoint whose referenced image bytes are unbound."
        )
    try:
        dataset_fingerprint = verify_dataset_fingerprint(a.table, expected_fingerprint)
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
    split_mode = ck.get("split_mode", "run")
    test_groups = ck.get("test_groups") or ck.get("test_runs")
    validation_groups = ck.get("validation_groups") or ck.get("validation_runs", [])
    if not test_groups:
        raise SystemExit(
            "checkpoint has no untouched test groups. Retrain with the current train_multimodal.py; "
            "validation results are intentionally not reported as test performance."
        )

    cols = ck["sensor_columns"]
    seq = int(ck.get("seq_len", 32))
    df = pd.read_csv(a.table)
    df = df.sort_values(["run_id", "ts"]).reset_index(drop=True)
    validate_modeling_table(df, cols)
    if split_mode == "spatial":
        block_m = float(ck.get("spatial_block_m") or 500.0)
        test = select_spatial_blocks(df, test_groups, block_size_m=block_m)
    else:
        test = select_runs(df, test_groups)
    ds = RailSequenceDataset(
        test,
        cols,
        seq_len=seq,
        sensor_mean=ck["sensor_mean"],
        sensor_std=ck["sensor_std"],
        image_mode=ck["image_mode"],
    )
    if len(ds) == 0:
        raise SystemExit("test split contains no complete forecasting windows")

    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionTransformer(sensor_dim=len(cols))
    model.load_state_dict(ck["state_dict"])
    model.to(device)

    collected = {}
    metrics = {}
    for mode in ("nonvisual_inputs_ablation", "visual_inputs_ablation", "multimodal"):
        collected[mode] = _collect(model, dl, device, mode, cols)
        metrics[mode] = _metrics(collected[mode])
    metrics = {"persistence": _persistence(collected["multimodal"]), **metrics}

    timing_offsets = [int(x) for x in a.timing_offset_steps.split(",") if x.strip()]
    dropout_probs = [float(x) for x in a.frame_dropout.split(",") if x.strip()]
    sensitivity = {"camera_offset_steps": {}, "frame_dropout": {}}
    for offset in timing_offsets:
        shifted = _collect(model, dl, device, "multimodal", cols, frame_offset_steps=offset, seed=a.sensitivity_seed)
        sensitivity["camera_offset_steps"][str(offset)] = _metrics(shifted)
    for probability in dropout_probs:
        if not 0.0 <= probability <= 1.0:
            raise SystemExit("--frame-dropout probabilities must be in [0,1]")
        dropped = _collect(model, dl, device, "multimodal", cols, frame_dropout=probability, seed=a.sensitivity_seed)
        sensitivity["frame_dropout"][f"{probability:.3f}"] = _metrics(dropped)

    group_column = "spatial_block_id" if split_mode == "spatial" else "run_id"
    per_group = {}
    for group in sorted(map(str, test_groups)):
        group_df = test[test[group_column].astype(str) == group].copy()
        group_ds = RailSequenceDataset(
            group_df, cols, seq_len=seq, sensor_mean=ck["sensor_mean"], sensor_std=ck["sensor_std"], image_mode=ck["image_mode"]
        )
        if len(group_ds) == 0:
            continue
        group_dl = DataLoader(group_ds, batch_size=a.batch, shuffle=False, num_workers=0)
        group_collected = _collect(model, group_dl, device, "multimodal", cols)
        per_group[group] = {
            "windows": len(group_ds),
            "multimodal": _metrics(group_collected),
            "persistence": _persistence(group_collected),
        }
    group_variability = {
        "per_group": per_group,
        "summary": summarize_group_variability(per_group),
    }

    report = {
        "split_mode": split_mode,
        "test_groups": sorted(map(str, test_groups)),
        "validation_groups": sorted(map(str, validation_groups)),
        "test_group_count": len(test_groups),
        "validation_group_count": len(validation_groups),
        "spatial_block_m": float(ck.get("spatial_block_m") or 0.0) if split_mode == "spatial" else None,
        "spatial_purge_margin_m": float(ck.get("spatial_purge_margin_m") or 0.0) if split_mode == "spatial" else None,
        "spatial_purged_train_rows": int(ck.get("spatial_purged_train_rows") or 0) if split_mode == "spatial" else 0,
        "spatial_purged_validation_rows": int(ck.get("spatial_purged_validation_rows") or 0) if split_mode == "spatial" else 0,
        "spatial_train_to_heldout_min_m": float(ck.get("spatial_train_to_heldout_min_m") or 0.0) if split_mode == "spatial" else None,
        "spatial_validation_to_test_min_m": float(ck.get("spatial_validation_to_test_min_m") or 0.0) if split_mode == "spatial" else None,
        "windows": len(ds),
        "horizons": list(HORIZONS),
        "sample_period_ms": 1000.0 * float(ck.get("sample_period_s", 0.1)),
        "model_arch_version": MODEL_ARCH_VERSION,
        "device": str(device),
        "dataset_fingerprint": dataset_fingerprint,
        "ablation_contract": {
            "nonvisual_inputs_ablation": [c for c in cols if c not in VISUAL_STRUCTURED_COLUMNS],
            "visual_inputs_ablation": ["camera_frames"] + [c for c in cols if c in VISUAL_STRUCTURED_COLUMNS],
            "multimodal": ["camera_frames"] + list(cols),
        },
        "metrics": metrics,
        "group_variability": group_variability,
        "sensitivity": sensitivity,
    }
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_md.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(report, indent=2) + "\n")
    a.out_md.write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
