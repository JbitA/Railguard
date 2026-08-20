from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(x: float) -> str:
    return f"{100.0*x:.1f}%"


def render(root: Path) -> str:
    summary = json.loads((root / "seed_summary.json").read_text())
    seeds = summary["training_seeds"]
    lines = [
        "# RailGuard-MM experiment results",
        "",
        f"Matched training seeds: **{', '.join(map(str, seeds))}**.",
        "",
        "| Model | Vibration MAE mean ± seed SD | Vision MAE mean ± seed SD |",
        "|---|---:|---:|",
    ]
    for model in ("persistence", "sensor_only", "vision_only", "multimodal"):
        vib = summary["model_variability"][model]["vibration_mae_mean"]
        vis = summary["model_variability"][model]["vision_mae_mean"]
        lines.append(f"| {model} | {vib['mean']:.5f} ± {vib['std']:.5f} | {vis['mean']:.5f} ± {vis['std']:.5f} |")
    lines += ["", "## Multimodal improvement across matched seeds", ""]
    for baseline in ("persistence", "sensor_only", "vision_only"):
        vib = summary["multimodal_error_reduction_by_seed"][baseline]["vibration_mae_mean"]
        lines.append(
            f"- vs **{baseline}**: vibration error reduction `{vib['mean_error_reduction']:.5f}` "
            f"(W/T/L {vib['wins']}/{vib['ties']}/{vib['losses']})."
        )
    lines += [
        "",
        "This card is generated from the experiment artifacts. Geographic uncertainty remains in each per-seed `independent_metrics.md`; seed variability and geographic bootstrap uncertainty are intentionally reported separately.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Render a README-ready result card from completed showcase experiments.")
    ap.add_argument("root", type=Path, nargs="?", default=Path("artifacts/showcase"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    text = render(args.root)
    out = args.out or args.root / "RESULTS.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
