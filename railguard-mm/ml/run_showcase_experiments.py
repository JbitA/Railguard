from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

MODALITIES = ("sensor_only", "vision_only", "multimodal")


def run(cmd: list[str], *, dry_run: bool) -> None:
    print("+", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run the portfolio experiment suite: classical baseline, matched-seed independent modalities, untouched-test evaluation, and robustness analysis."
    )
    ap.add_argument("table", type=Path)
    ap.add_argument("--seeds", default="7,17,37", help="comma-separated matched training seeds")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--split-seed", type=int, default=7)
    ap.add_argument("--spatial-block-m", type=float, default=500.0)
    ap.add_argument("--purge-margin-m", type=float, default=30.0)
    ap.add_argument("--visual-init", type=Path, default=None, help="optional rail-surface visual-pretraining checkpoint; applied to vision_only and multimodal")
    ap.add_argument("--out", type=Path, default=Path("artifacts/showcase"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    if len(set(seeds)) != len(seeds) or not seeds:
        raise SystemExit("--seeds must contain one or more unique integers")
    args.out.mkdir(parents=True, exist_ok=True)
    python = sys.executable

    manifest = {
        "suite_version": 1,
        "table": str(args.table),
        "seeds": seeds,
        "split_seed": args.split_seed,
        "spatial_block_m": args.spatial_block_m,
        "purge_margin_m": args.purge_margin_m,
        "visual_pretraining": str(args.visual_init) if args.visual_init else None,
        "experiments": [
            "classical persistence/random-forest baseline",
            "independently trained sensor-only vs vision-only vs multimodal",
            "untouched-spatial-test held-out-group bootstrap uncertainty",
            "camera timing-offset sensitivity",
            "camera frame-dropout sensitivity",
            "matched-seed variability summary",
        ],
    }
    (args.out / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    classical_dir = args.out / "classical"
    run([
        python, "ml/train_classical.py", str(args.table),
        "--out", str(classical_dir / "models"),
        "--metrics", str(classical_dir / "metrics.json"),
        "--split-seed", str(args.split_seed),
        "--seed", str(seeds[0]),
        "--spatial-block-m", str(args.spatial_block_m),
        "--spatial-purge-margin-m", str(args.purge_margin_m),
    ], dry_run=args.dry_run)

    reports: list[Path] = []
    for seed in seeds:
        seed_dir = args.out / f"seed_{seed}"
        checkpoints: dict[str, Path] = {}
        for modality in MODALITIES:
            ck = seed_dir / f"{modality}.pt"
            checkpoints[modality] = ck
            cmd = [
                python, "ml/train_multimodal.py", str(args.table),
                "--epochs", str(args.epochs), "--batch", str(args.batch),
                "--split-seed", str(args.split_seed), "--seed", str(seed),
                "--spatial-block-m", str(args.spatial_block_m),
                "--spatial-purge-margin-m", str(args.purge_margin_m),
                "--modality", modality, "--out", str(ck),
            ]
            if args.visual_init is not None and modality != "sensor_only":
                cmd += ["--frame-encoder-init", str(args.visual_init)]
            run(cmd, dry_run=args.dry_run)

        report_json = seed_dir / "independent_metrics.json"
        report_md = seed_dir / "independent_metrics.md"
        reports.append(report_json)
        run([
            python, "ml/evaluate_independent_models.py", str(args.table),
            "--sensor-only", str(checkpoints["sensor_only"]),
            "--vision-only", str(checkpoints["vision_only"]),
            "--multimodal", str(checkpoints["multimodal"]),
            "--batch", str(args.batch),
            "--out-json", str(report_json), "--out-md", str(report_md),
        ], dry_run=args.dry_run)
        run([
            python, "ml/evaluate_multimodal.py", str(args.table), str(checkpoints["multimodal"]),
            "--batch", str(args.batch),
            "--out-json", str(seed_dir / "robustness.json"),
            "--out-md", str(seed_dir / "robustness.md"),
        ], dry_run=args.dry_run)

    run([
        python, "ml/summarize_seed_runs.py", *map(str, reports),
        "--out-json", str(args.out / "seed_summary.json"),
        "--out-md", str(args.out / "seed_summary.md"),
    ], dry_run=args.dry_run)
    run([python, "scripts/render_showcase_results.py", str(args.out)], dry_run=args.dry_run)

    print(f"showcase suite {'plan written' if args.dry_run else 'completed'} under {args.out}")


if __name__ == "__main__":
    main()
