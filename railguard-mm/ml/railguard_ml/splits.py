from __future__ import annotations

from dataclasses import dataclass
import random
import pandas as pd


@dataclass(frozen=True)
class GroupSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_runs: list[str]
    validation_runs: list[str]
    test_runs: list[str]


def _fraction_count(n: int, fraction: float) -> int:
    if fraction <= 0:
        return 0
    return max(1, int(round(n * fraction)))


def grouped_train_val_test_split(
    df: pd.DataFrame,
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
    seed: int = 7,
) -> GroupSplit:
    """Deterministic run-disjoint train/validation/test split.

    A true test split is intentionally kept untouched by checkpoint selection.  For
    Rail-VIVID, where multiple physical passes are available, rows from one run may
    only appear in one partition.  A single-run table is rejected: reporting a
    random-row test score would create severe temporal/location leakage.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be >0 and sum to <1")
    if "run_id" not in df.columns:
        raise ValueError("run_id is required for leakage-safe model evaluation")

    runs = list(dict.fromkeys(df["run_id"].astype(str).tolist()))
    if len(runs) < 3:
        raise ValueError("at least three physical runs are required for train/validation/test separation")

    shuffled = runs.copy()
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = _fraction_count(n, test_fraction)
    n_val = _fraction_count(n, val_fraction)
    # Always preserve at least one training run.
    while n_test + n_val >= n:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError("not enough physical runs for the requested split fractions")

    test_runs = sorted(shuffled[:n_test])
    validation_runs = sorted(shuffled[n_test:n_test + n_val])
    train_runs = sorted(shuffled[n_test + n_val:])

    run_series = df["run_id"].astype(str)
    train = df[run_series.isin(train_runs)].copy()
    validation = df[run_series.isin(validation_runs)].copy()
    test = df[run_series.isin(test_runs)].copy()

    return GroupSplit(train, validation, test, train_runs, validation_runs, test_runs)


def select_runs(df: pd.DataFrame, runs: list[str]) -> pd.DataFrame:
    if "run_id" not in df.columns:
        raise ValueError("run_id is required")
    wanted = set(map(str, runs))
    return df[df["run_id"].astype(str).isin(wanted)].copy()


@dataclass(frozen=True)
class SpatialSplit:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    train_blocks: list[str]
    validation_blocks: list[str]
    test_blocks: list[str]
    block_size_m: float
    purge_margin_m: float
    purged_train_rows: int
    purged_validation_rows: int
    train_to_heldout_min_m: float
    validation_to_test_min_m: float


def add_spatial_blocks(df: pd.DataFrame, *, block_size_m: float = 500.0) -> pd.DataFrame:
    """Attach stable Web-Mercator spatial blocks and contiguous sequence groups.

    The block origin is global rather than dataset-relative, so saved block IDs are
    reproducible when a table is reloaded.  `sequence_group_id` also increments when
    a physical run crosses a block boundary, preventing a temporal training window
    from jumping across a removed/held-out geographic segment after filtering.
    """
    import math
    import numpy as np

    if block_size_m <= 0:
        raise ValueError("block_size_m must be positive")
    if not {"latitude", "longitude", "run_id"}.issubset(df.columns):
        raise ValueError("spatial splitting requires run_id, latitude and longitude")
    out = df.copy()
    # Preserve the position in the fully sorted modeling table.  This lets the
    # split code detect gaps after geographic purging and prevents a temporal
    # sequence from being stitched across rows that were deliberately removed.
    out["_source_pos"] = range(len(out))
    lat = pd.to_numeric(out["latitude"], errors="coerce").to_numpy(float)
    lon = pd.to_numeric(out["longitude"], errors="coerce").to_numpy(float)
    if not (np.isfinite(lat).all() and np.isfinite(lon).all()):
        raise ValueError("spatial splitting requires finite latitude/longitude for every modeling row")
    if ((lat < -85.0) | (lat > 85.0) | (lon < -180.0) | (lon > 180.0)).any():
        raise ValueError("latitude/longitude outside supported geographic bounds")

    radius = 6_378_137.0
    x = radius * np.radians(lon)
    y = radius * np.log(np.tan(math.pi / 4.0 + np.radians(lat) / 2.0))
    bx = np.floor(x / block_size_m).astype(np.int64)
    by = np.floor(y / block_size_m).astype(np.int64)
    blocks = [f"{a}:{b}" for a, b in zip(bx, by)]
    out["spatial_block_id"] = blocks

    sequence_groups = [""] * len(out)
    # Preserve the current table ordering within each run; training scripts sort by
    # run/time before calling this helper.
    for run, idxs in out.groupby("run_id", sort=False).groups.items():
        previous = None
        segment = -1
        for idx in list(idxs):
            block = out.at[idx, "spatial_block_id"]
            if block != previous:
                segment += 1
                previous = block
            sequence_groups[out.index.get_loc(idx)] = f"{run}|{block}|segment-{segment}"
    out["sequence_group_id"] = sequence_groups
    return out


def _purge_near_reference(rows: pd.DataFrame, reference: pd.DataFrame, margin_m: float) -> tuple[pd.DataFrame, int]:
    """Remove rows within ``margin_m`` of any reference row using haversine distance."""
    import numpy as np
    from sklearn.neighbors import BallTree

    if margin_m <= 0 or rows.empty or reference.empty:
        return rows.copy(), 0
    earth_radius_m = 6_371_000.0
    ref_coords = np.radians(reference[["latitude", "longitude"]].to_numpy(float))
    row_coords = np.radians(rows[["latitude", "longitude"]].to_numpy(float))
    tree = BallTree(ref_coords, metric="haversine")
    nearest_rad, _ = tree.query(row_coords, k=1)
    keep = nearest_rad[:, 0] * earth_radius_m >= margin_m
    kept = rows.loc[keep].copy()
    return kept, int((~keep).sum())




def minimum_geodesic_distance_m(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Minimum row-to-row haversine distance between two non-empty partitions."""
    import numpy as np
    from sklearn.neighbors import BallTree

    if a.empty or b.empty:
        return float("inf")
    earth_radius_m = 6_371_000.0
    b_coords = np.radians(b[["latitude", "longitude"]].to_numpy(float))
    a_coords = np.radians(a[["latitude", "longitude"]].to_numpy(float))
    tree = BallTree(b_coords, metric="haversine")
    nearest_rad, _ = tree.query(a_coords, k=1)
    return float(nearest_rad[:, 0].min() * earth_radius_m)

def _resegment_after_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Rebuild sequence IDs so filtering cannot create artificial temporal continuity."""
    if df.empty:
        return df.copy()
    out = df.sort_values("_source_pos").copy()
    sequence_ids: dict[object, str] = {}
    for run, group in out.groupby("run_id", sort=False):
        previous_block = None
        previous_pos = None
        segment = -1
        for idx, row in group.iterrows():
            block = str(row["spatial_block_id"])
            pos = int(row["_source_pos"])
            if block != previous_block or previous_pos is None or pos != previous_pos + 1:
                segment += 1
            sequence_ids[idx] = f"{run}|{block}|segment-{segment}"
            previous_block = block
            previous_pos = pos
    out["sequence_group_id"] = pd.Series(sequence_ids)
    return out


def reset_partition_boundary_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Neutralize one-step features whose source observation lies outside a segment.

    ``vision_motion`` and derived ``speed_mps`` are calculated during preprocessing
    from the previous time bin.  After a spatial holdout/purge, the first retained
    row of a contiguous segment may therefore still contain a value computed using
    an observation that belongs to another partition or a purged gap.  Resetting
    exactly those boundary rows removes that cross-partition dependency without
    recomputing every image feature.
    """
    if df.empty or "sequence_group_id" not in df.columns:
        return df.copy()
    out = df.copy()
    first_indices = out.groupby("sequence_group_id", sort=False).head(1).index
    for column in ("vision_motion", "speed_mps"):
        if column in out.columns:
            out.loc[first_indices, column] = 0.0
    out["temporal_boundary_reset"] = False
    out.loc[first_indices, "temporal_boundary_reset"] = True
    return out


def spatial_train_val_test_split(
    df: pd.DataFrame,
    *,
    block_size_m: float = 500.0,
    val_fraction: float = 0.15,
    test_fraction: float = 0.20,
    seed: int = 7,
    purge_margin_m: float = 30.0,
) -> SpatialSplit:
    """Split whole geographic blocks across every physical pass.

    This is stricter than a run-only split for repeated-route datasets: imagery and
    vibration from a test block never occur in a training pass of the same location.
    A metric purge margin further removes train rows close to validation/test
    geography and validation rows close to test geography, avoiding arbitrarily
    near samples on opposite sides of a block boundary.
    """
    if not 0 < val_fraction < 1 or not 0 < test_fraction < 1 or val_fraction + test_fraction >= 1:
        raise ValueError("val_fraction and test_fraction must be >0 and sum to <1")
    if purge_margin_m < 0:
        raise ValueError("purge_margin_m must be nonnegative")
    enriched = add_spatial_blocks(df, block_size_m=block_size_m)
    blocks = list(dict.fromkeys(enriched["spatial_block_id"].tolist()))
    if len(blocks) < 3:
        raise ValueError("at least three spatial blocks are required for spatial train/validation/test separation")

    shuffled = blocks.copy()
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = _fraction_count(n, test_fraction)
    n_val = _fraction_count(n, val_fraction)
    while n_test + n_val >= n:
        if n_test >= n_val and n_test > 1:
            n_test -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            raise ValueError("not enough spatial blocks for the requested fractions")

    test_blocks = sorted(shuffled[:n_test])
    validation_blocks = sorted(shuffled[n_test:n_test + n_val])
    train_blocks = sorted(shuffled[n_test + n_val:])
    groups = enriched["spatial_block_id"]
    train = enriched[groups.isin(train_blocks)].copy()
    validation = enriched[groups.isin(validation_blocks)].copy()
    test = enriched[groups.isin(test_blocks)].copy()

    # Preserve the test geography exactly.  Training is purged around both
    # checkpoint-selection and test geography; validation is purged around test.
    train, purged_train = _purge_near_reference(
        train, pd.concat([validation, test], ignore_index=True), float(purge_margin_m)
    )
    validation, purged_validation = _purge_near_reference(validation, test, float(purge_margin_m))
    train = _resegment_after_filter(train)
    validation = _resegment_after_filter(validation)
    test = _resegment_after_filter(test)
    train = reset_partition_boundary_temporal_features(train)
    validation = reset_partition_boundary_temporal_features(validation)
    test = reset_partition_boundary_temporal_features(test)
    if train.empty or validation.empty or test.empty:
        raise ValueError("spatial purge margin removed an entire split; reduce --spatial-purge-margin-m")

    return SpatialSplit(
        train,
        validation,
        test,
        train_blocks,
        validation_blocks,
        test_blocks,
        float(block_size_m),
        float(purge_margin_m),
        purged_train,
        purged_validation,
        minimum_geodesic_distance_m(train, pd.concat([validation, test], ignore_index=True)),
        minimum_geodesic_distance_m(validation, test),
    )


def select_spatial_blocks(df: pd.DataFrame, blocks: list[str], *, block_size_m: float) -> pd.DataFrame:
    enriched = add_spatial_blocks(df, block_size_m=block_size_m)
    wanted = set(map(str, blocks))
    selected = enriched[enriched["spatial_block_id"].astype(str).isin(wanted)].copy()
    selected = _resegment_after_filter(selected)
    return reset_partition_boundary_temporal_features(selected)
