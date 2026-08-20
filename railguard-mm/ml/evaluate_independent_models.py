from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from railguard_ml.dataset import RailSequenceDataset
    from railguard_ml.models import FusionTransformer
    from railguard_ml.model_comparison import validate_independent_checkpoint_set
    from railguard_ml.provenance import verify_dataset_fingerprint
    from railguard_ml.splits import select_runs, select_spatial_blocks
    from railguard_ml.statistics import paired_group_bootstrap
    from railguard_ml.validation import validate_modeling_table
    from evaluate_multimodal import _collect, _metrics, _persistence
except ModuleNotFoundError:
    from ml.railguard_ml.dataset import RailSequenceDataset
    from ml.railguard_ml.models import FusionTransformer
    from ml.railguard_ml.model_comparison import validate_independent_checkpoint_set
    from ml.railguard_ml.provenance import verify_dataset_fingerprint
    from ml.railguard_ml.splits import select_runs, select_spatial_blocks
    from ml.railguard_ml.statistics import paired_group_bootstrap
    from ml.railguard_ml.validation import validate_modeling_table
    from ml.evaluate_multimodal import _collect, _metrics, _persistence


NEURAL_MODALITIES = ("sensor_only", "vision_only", "multimodal")
ALL_MODELS = ("persistence",) + NEURAL_MODALITIES


def _markdown(report: dict) -> str:
    lines = [
        "# Independently trained modality comparison",
        "",
        "Each neural row is a separately optimized checkpoint trained with the same data/split contract; this is distinct from masking modalities after a fusion model has already been trained.",
        "",
        "| Model | Vib MAE +1 | +5 | +10 | Vision MAE +1 | +5 | +10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ALL_MODELS:
        m = report["metrics"][name]
        v, vis = m["vibration_mae"], m["vision_mae"]
        lines.append(
            f"| {name} | {v[0]:.5f} | {v[1]:.5f} | {v[2]:.5f} | "
            f"{vis[0]:.5f} | {vis[1]:.5f} | {vis[2]:.5f} |"
        )
    lines += [
        "",
        f"Split: **{report['contract']['split_mode']}**; untouched groups: **{len(report['contract']['test_groups'])}**; windows: **{report['windows']}**.",
        "",
    ]

    paired = report.get("paired_group_uncertainty", {})
    if paired:
        lines += [
            "## Paired held-out-group uncertainty",
            "",
            "Positive error reduction means multimodal has lower error. Confidence intervals resample whole held-out geographic blocks/runs, not individual autocorrelated time windows.",
            "",
            "| Baseline | Vib MAE reduction | 95% CI | P(reduction > 0) | Vision MAE reduction | 95% CI | P(reduction > 0) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for baseline in ("persistence", "sensor_only", "vision_only"):
            item = paired[baseline]
            vib = item["vibration_mae_mean"]
            vis = item["vision_mae_mean"]

            def fmt_ci(x: dict) -> tuple[str, str, str]:
                estimate = f"{x['estimate']:.5f}"
                if x["ci95_low"] is None:
                    return estimate, "n/a (<3 groups)", "n/a"
                ci = f"[{x['ci95_low']:.5f}, {x['ci95_high']:.5f}]"
                prob = f"{x['probability_improvement']:.1%}"
                return estimate, ci, prob

            ve, vci, vp = fmt_ci(vib)
            ie, ici, ip = fmt_ci(vis)
            lines.append(f"| {baseline} | {ve} | {vci} | {vp} | {ie} | {ici} | {ip} |")
        lines += [
            "",
            f"Bootstrap repetitions: **{report['bootstrap']['repeats']}**; seed: **{report['bootstrap']['seed']}**. "
            "These intervals quantify between-held-out-group uncertainty; they are not IID-window confidence intervals or a substitute for external replication.",
            "",
        ]
    return "\n".join(lines)


def _dataset_for_group(test: pd.DataFrame, contract: dict, group: str) -> pd.DataFrame:
    if contract["split_mode"] == "spatial":
        column = "spatial_block_id"
    else:
        column = "run_id"
    return test[test[column].astype(str) == str(group)].copy()


def _build_model(checkpoint: dict, sensor_dim: int, device: torch.device) -> FusionTransformer:
    model = FusionTransformer(sensor_dim=sensor_dim)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model


def _group_metrics(
    test: pd.DataFrame,
    contract: dict,
    checkpoints: dict[str, dict],
    models: dict[str, FusionTransformer],
    cols: list[str],
    device: torch.device,
    batch: int,
) -> dict[str, dict]:
    per_group: dict[str, dict] = {}
    for group in contract["test_groups"]:
        group_df = _dataset_for_group(test, contract, str(group))
        group_result: dict[str, dict] = {}
        persistence_source = None
        windows = None
        for modality in NEURAL_MODALITIES:
            ck = checkpoints[modality]
            ds = RailSequenceDataset(
                group_df,
                cols,
                seq_len=int(ck["seq_len"]),
                sensor_mean=ck["sensor_mean"],
                sensor_std=ck["sensor_std"],
                image_mode=ck["image_mode"],
            )
            if len(ds) == 0:
                continue
            if windows is None:
                windows = len(ds)
            elif len(ds) != windows:
                raise ValueError(f"checkpoint window mismatch inside held-out group {group}")
            dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
            collected = _collect(models[modality], dl, device, modality, cols)
            group_result[modality] = _metrics(collected)
            if persistence_source is None:
                persistence_source = collected
        if persistence_source is None or windows is None or len(group_result) != len(NEURAL_MODALITIES):
            continue
        group_result["persistence"] = _persistence(persistence_source)
        group_result["windows"] = windows
        per_group[str(group)] = group_result
    return per_group


def paired_uncertainty(per_group: dict[str, dict], *, repeats: int, seed: int) -> dict:
    if not per_group:
        return {}
    out: dict[str, dict] = {}
    for baseline in ("persistence", "sensor_only", "vision_only"):
        out[baseline] = {}
        for metric in ("vibration_mae_mean", "vision_mae_mean"):
            baseline_values = {g: float(v[baseline][metric]) for g, v in per_group.items()}
            candidate_values = {g: float(v["multimodal"][metric]) for g, v in per_group.items()}
            out[baseline][metric] = paired_group_bootstrap(
                baseline_values, candidate_values, repeats=repeats, seed=seed
            ).to_dict()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare independently trained sensor-only, vision-only and multimodal checkpoints on one untouched split.")
    ap.add_argument("table", type=Path)
    ap.add_argument("--sensor-only", type=Path, required=True)
    ap.add_argument("--vision-only", type=Path, required=True)
    ap.add_argument("--multimodal", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--bootstrap-repeats", type=int, default=5000, help="paired held-out-group bootstrap repetitions")
    ap.add_argument("--bootstrap-seed", type=int, default=17)
    ap.add_argument("--out-json", type=Path, default=Path("artifacts/evaluation/independent_model_metrics.json"))
    ap.add_argument("--out-md", type=Path, default=Path("artifacts/evaluation/independent_model_metrics.md"))
    a = ap.parse_args()
    if a.bootstrap_repeats <= 0:
        raise SystemExit("--bootstrap-repeats must be positive")

    paths = {"sensor_only": a.sensor_only, "vision_only": a.vision_only, "multimodal": a.multimodal}
    checkpoints = {name: torch.load(path, map_location="cpu", weights_only=False) for name, path in paths.items()}
    try:
        contract = validate_independent_checkpoint_set(checkpoints)
        for ck in checkpoints.values():
            verify_dataset_fingerprint(a.table, ck["provenance"]["dataset_fingerprint"])
    except (ValueError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc

    cols = list(checkpoints["multimodal"]["sensor_columns"])
    df = pd.read_csv(a.table).sort_values(["run_id", "ts"]).reset_index(drop=True)
    validate_modeling_table(df, cols)
    if contract["split_mode"] == "spatial":
        test = select_spatial_blocks(df, contract["test_groups"], block_size_m=float(contract["spatial_block_m"]))
    else:
        test = select_runs(df, contract["test_groups"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = {name: _build_model(checkpoints[name], len(cols), device) for name in NEURAL_MODALITIES}
    metrics: dict[str, dict] = {}
    persistence_source = None
    window_count = None
    for modality in NEURAL_MODALITIES:
        ck = checkpoints[modality]
        ds = RailSequenceDataset(test, cols, seq_len=int(ck["seq_len"]), sensor_mean=ck["sensor_mean"], sensor_std=ck["sensor_std"], image_mode=ck["image_mode"])
        if len(ds) == 0:
            raise SystemExit("untouched test split contains no complete forecasting windows")
        if window_count is None:
            window_count = len(ds)
        elif len(ds) != window_count:
            raise SystemExit("independent checkpoints produced different test-window counts")
        dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0)
        collected = _collect(models[modality], dl, device, modality, cols)
        metrics[modality] = _metrics(collected)
        if persistence_source is None:
            persistence_source = collected
    assert persistence_source is not None and window_count is not None
    metrics = {"persistence": _persistence(persistence_source), **metrics}

    try:
        per_group = _group_metrics(test, contract, checkpoints, models, cols, device, a.batch)
        uncertainty = paired_uncertainty(per_group, repeats=a.bootstrap_repeats, seed=a.bootstrap_seed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    report = {
        "contract": contract,
        "windows": window_count,
        "device": str(device),
        "checkpoints": {name: str(path) for name, path in paths.items()},
        "metrics": metrics,
        "per_group": per_group,
        "paired_group_uncertainty": uncertainty,
        "bootstrap": {"repeats": a.bootstrap_repeats, "seed": a.bootstrap_seed, "unit": "held-out group"},
    }
    a.out_json.parent.mkdir(parents=True, exist_ok=True)
    a.out_md.parent.mkdir(parents=True, exist_ok=True)
    a.out_json.write_text(json.dumps(report, indent=2) + "\n")
    a.out_md.write_text(_markdown(report))
    print(_markdown(report))


if __name__ == "__main__":
    main()
