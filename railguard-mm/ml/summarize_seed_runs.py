from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

MODELS = ("persistence", "sensor_only", "vision_only", "multimodal")
METRICS = ("vibration_mae_mean", "vision_mae_mean")


def _base_contract(contract: dict) -> dict:
    return {k: v for k, v in contract.items() if k != "training_seed"}


def summarize_reports(reports: list[dict]) -> dict:
    if not reports:
        raise ValueError("at least one independent-model report is required")
    reference = _base_contract(reports[0]["contract"])
    seeds: list[int] = []
    for report in reports:
        contract = report.get("contract") or {}
        if _base_contract(contract) != reference:
            raise ValueError("seed reports do not share the same dataset/split/model experiment contract")
        seed = contract.get("training_seed")
        if seed is None:
            raise ValueError("seed report is missing training_seed")
        seeds.append(int(seed))
        if any(model not in report.get("metrics", {}) for model in MODELS):
            raise ValueError("seed report is missing required model metrics")
    if len(set(seeds)) != len(seeds):
        raise ValueError("training seeds must be unique across repeated experiment reports")

    model_stats: dict[str, dict] = {}
    for model in MODELS:
        model_stats[model] = {}
        for metric in METRICS:
            values = np.asarray([float(r["metrics"][model][metric]) for r in reports], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError("seed metrics must be finite")
            model_stats[model][metric] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "min": float(values.min()),
                "max": float(values.max()),
            }

    improvements: dict[str, dict] = {}
    for baseline in ("persistence", "sensor_only", "vision_only"):
        improvements[baseline] = {}
        for metric in METRICS:
            deltas = np.asarray([
                float(r["metrics"][baseline][metric]) - float(r["metrics"]["multimodal"][metric])
                for r in reports
            ])
            improvements[baseline][metric] = {
                "mean_error_reduction": float(deltas.mean()),
                "std": float(deltas.std(ddof=1)) if deltas.size > 1 else 0.0,
                "wins": int((deltas > 0).sum()),
                "ties": int((deltas == 0).sum()),
                "losses": int((deltas < 0).sum()),
            }
    return {
        "experiment_contract": reference,
        "training_seeds": sorted(seeds),
        "seed_count": len(seeds),
        "model_variability": model_stats,
        "multimodal_error_reduction_by_seed": improvements,
    }


def markdown(summary: dict) -> str:
    lines = [
        "# Repeated-seed modality experiment",
        "",
        f"Training seeds: **{summary['seed_count']}** — `{', '.join(map(str, summary['training_seeds']))}`.",
        "",
        "These are descriptive training-seed variability statistics on an identical untouched test contract. They do not treat seeds as independent geographic test samples and do not replace the held-out-group bootstrap in each per-seed report.",
        "",
        "| Model | Vibration MAE mean ± seed SD | Vision MAE mean ± seed SD |",
        "|---|---:|---:|",
    ]
    for model in MODELS:
        vib = summary["model_variability"][model]["vibration_mae_mean"]
        vis = summary["model_variability"][model]["vision_mae_mean"]
        lines.append(f"| {model} | {vib['mean']:.5f} ± {vib['std']:.5f} | {vis['mean']:.5f} ± {vis['std']:.5f} |")
    lines += ["", "## Multimodal paired seed outcomes", "", "| Baseline | Vib reduction mean ± SD | W/T/L | Vision reduction mean ± SD | W/T/L |", "|---|---:|---:|---:|---:|"]
    for baseline in ("persistence", "sensor_only", "vision_only"):
        vib = summary["multimodal_error_reduction_by_seed"][baseline]["vibration_mae_mean"]
        vis = summary["multimodal_error_reduction_by_seed"][baseline]["vision_mae_mean"]
        lines.append(
            f"| {baseline} | {vib['mean_error_reduction']:.5f} ± {vib['std']:.5f} | {vib['wins']}/{vib['ties']}/{vib['losses']} | "
            f"{vis['mean_error_reduction']:.5f} ± {vis['std']:.5f} | {vis['wins']}/{vis['ties']}/{vis['losses']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize independently trained modality comparisons across matched random seeds.")
    ap.add_argument("reports", nargs="+", type=Path)
    ap.add_argument("--out-json", type=Path, default=Path("artifacts/evaluation/seed_summary.json"))
    ap.add_argument("--out-md", type=Path, default=Path("artifacts/evaluation/seed_summary.md"))
    args = ap.parse_args()
    reports = [json.loads(p.read_text()) for p in args.reports]
    try:
        summary = summarize_reports(reports)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2) + "\n")
    args.out_md.write_text(markdown(summary))
    print(markdown(summary))


if __name__ == "__main__":
    main()
