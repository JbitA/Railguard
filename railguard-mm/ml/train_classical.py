from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from railguard_ml.contracts import DEPLOYMENT_SENSOR_COLUMNS
    from railguard_ml.provenance import training_provenance
    from railguard_ml.validation import validate_modeling_table
    from railguard_ml.splits import (
        add_spatial_blocks,
        grouped_train_val_test_split,
        spatial_train_val_test_split,
    )
except ModuleNotFoundError:
    from ml.railguard_ml.contracts import DEPLOYMENT_SENSOR_COLUMNS
    from ml.railguard_ml.provenance import training_provenance
    from ml.railguard_ml.validation import validate_modeling_table
    from ml.railguard_ml.splits import (
        add_spatial_blocks,
        grouped_train_val_test_split,
        spatial_train_val_test_split,
    )

FEATURES = DEPLOYMENT_SENSOR_COLUMNS
HORIZONS = (1, 5, 10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-safe classical RailGuard baselines")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--metrics", type=Path, default=Path("artifacts/evaluation/classical_metrics.json"))
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--seed", type=int, default=17, help="estimator random seed")
    parser.add_argument("--split-mode", choices=["spatial", "run"], default="spatial")
    parser.add_argument("--spatial-block-m", type=float, default=500.0)
    parser.add_argument("--spatial-purge-margin-m", type=float, default=30.0)
    return parser.parse_args()


def add_forecast_targets(df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby(group_column, sort=False)
    for horizon in HORIZONS:
        out[f"target_{horizon}"] = grouped["vibration_rms"].shift(-horizon)
    return out


def main() -> None:
    args = parse_args()
    provenance = training_provenance(args.csv, seed=args.seed, deterministic=True)
    df = pd.read_csv(args.csv).sort_values(["run_id", "ts"]).reset_index(drop=True)
    validate_modeling_table(df, FEATURES)

    # Split and apply any geographic purge before constructing future targets.
    # Otherwise a retained input row could still carry a target copied from a
    # row that was subsequently purged near held-out geography.
    base = df.reset_index(drop=True)
    if args.split_mode == "spatial":
        split = spatial_train_val_test_split(
            base,
            block_size_m=args.spatial_block_m,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.split_seed,
            purge_margin_m=args.spatial_purge_margin_m,
        )
        validation_groups, test_groups = split.validation_blocks, split.test_blocks
        group_column = "sequence_group_id"
    else:
        split = grouped_train_val_test_split(
            base,
            val_fraction=args.val_fraction,
            test_fraction=args.test_fraction,
            seed=args.split_seed,
        )
        validation_groups, test_groups = split.validation_runs, split.test_runs
        group_column = "run_id"

    target_columns = [f"target_{h}" for h in HORIZONS]
    def finalize_partition(part: pd.DataFrame) -> pd.DataFrame:
        return add_forecast_targets(part, group_column).dropna(subset=FEATURES + target_columns).reset_index(drop=True)

    train = finalize_partition(split.train)
    validation = finalize_partition(split.validation)
    test = finalize_partition(split.test)
    if train.empty or validation.empty or test.empty:
        raise SystemExit("split/purge left insufficient rows for +1/+5/+10 classical forecast targets")
    regressor = RandomForestRegressor(
        n_estimators=400,
        min_samples_leaf=3,
        n_jobs=-1,
        random_state=args.seed,
    )
    regressor.fit(train[FEATURES], train[target_columns])

    prediction = regressor.predict(test[FEATURES])
    truth = test[target_columns].to_numpy()
    random_forest_mae = [mean_absolute_error(truth[:, i], prediction[:, i]) for i in range(3)]

    persistence = np.repeat(test["vibration_rms"].to_numpy()[:, None], 3, axis=1)
    persistence_mae = [mean_absolute_error(truth[:, i], persistence[:, i]) for i in range(3)]

    isolation_forest = Pipeline(
        [
            ("scale", StandardScaler()),
            ("model", IsolationForest(n_estimators=300, contamination="auto", random_state=args.seed)),
        ]
    )
    isolation_forest.fit(train[FEATURES])

    args.out.mkdir(parents=True, exist_ok=True)
    joblib.dump(regressor, args.out / "vibration_rf.joblib")
    joblib.dump(isolation_forest, args.out / "multimodal_isolation_forest.joblib")

    report = {
        "split_mode": args.split_mode,
        "spatial_block_m": args.spatial_block_m if args.split_mode == "spatial" else None,
        "spatial_purge_margin_m": args.spatial_purge_margin_m if args.split_mode == "spatial" else None,
        "spatial_purged_train_rows": split.purged_train_rows if args.split_mode == "spatial" else 0,
        "spatial_purged_validation_rows": split.purged_validation_rows if args.split_mode == "spatial" else 0,
        "spatial_train_to_heldout_min_m": split.train_to_heldout_min_m if args.split_mode == "spatial" else None,
        "spatial_validation_to_test_min_m": split.validation_to_test_min_m if args.split_mode == "spatial" else None,
        "validation_groups": validation_groups,
        "test_groups": test_groups,
        "split_seed": args.split_seed,
        "training_seed": args.seed,
        "provenance": provenance,
        "horizons": list(HORIZONS),
        "random_forest_vibration_mae": random_forest_mae,
        "persistence_vibration_mae": persistence_mae,
        "improvement_percent": [
            float(100.0 * (baseline - model) / (baseline + 1e-12))
            for baseline, model in zip(persistence_mae, random_forest_mae)
        ],
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
