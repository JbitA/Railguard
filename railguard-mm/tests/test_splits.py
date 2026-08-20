import pandas as pd
import pytest

from ml.railguard_ml.splits import grouped_train_val_test_split


def _table(n_runs=10):
    return pd.DataFrame({
        "run_id": [f"run-{r}" for r in range(n_runs) for _ in range(5)],
        "x": list(range(n_runs * 5)),
    })


def test_group_split_is_run_disjoint_and_reproducible():
    df = _table()
    a = grouped_train_val_test_split(df, seed=13)
    b = grouped_train_val_test_split(df, seed=13)
    assert a.train_runs == b.train_runs
    assert a.validation_runs == b.validation_runs
    assert a.test_runs == b.test_runs
    assert set(a.train_runs).isdisjoint(a.validation_runs)
    assert set(a.train_runs).isdisjoint(a.test_runs)
    assert set(a.validation_runs).isdisjoint(a.test_runs)
    assert len(a.train) + len(a.validation) + len(a.test) == len(df)


def test_group_split_refuses_single_or_two_run_fake_test_split():
    with pytest.raises(ValueError):
        grouped_train_val_test_split(_table(2))

from ml.railguard_ml.splits import add_spatial_blocks, spatial_train_val_test_split, reset_partition_boundary_temporal_features, select_spatial_blocks


def _spatial_table(n_blocks=12, rows_per_block=20):
    rows=[]
    # ~0.01 degrees longitude is ~850m at this latitude, comfortably >500m blocks.
    for run in range(3):
        for block in range(n_blocks):
            for i in range(rows_per_block):
                rows.append({
                    "run_id":f"run-{run}",
                    "ts":f"2026-01-01T00:{block:02d}:{i:02d}Z",
                    "latitude":40.0,
                    "longitude":-77.0 + block*0.01,
                })
    return pd.DataFrame(rows)


def test_spatial_split_keeps_locations_disjoint_across_repeated_runs():
    df=_spatial_table()
    parts=spatial_train_val_test_split(df,block_size_m=500,seed=9)
    assert set(parts.train_blocks).isdisjoint(parts.validation_blocks)
    assert set(parts.train_blocks).isdisjoint(parts.test_blocks)
    assert set(parts.validation_blocks).isdisjoint(parts.test_blocks)
    assert set(parts.train.spatial_block_id).isdisjoint(set(parts.test.spatial_block_id))
    # The same physical runs may occur in all partitions; location, not run identity, is held out.
    assert set(parts.train.run_id) == set(parts.test.run_id)
    assert "sequence_group_id" in parts.train.columns


def test_spatial_sequence_group_changes_at_block_boundary():
    enriched=add_spatial_blocks(_spatial_table(n_blocks=3,rows_per_block=2),block_size_m=500)
    run=enriched[enriched.run_id=="run-0"]
    assert run.sequence_group_id.nunique() >= 3


def _continuous_route_table(n_runs=3, n_points=240, spacing_m=10.0):
    # At latitude 40 deg, one longitude degree is roughly 85.3 km.
    lon_step = spacing_m / 85_300.0
    rows=[]
    for run in range(n_runs):
        for i in range(n_points):
            rows.append({
                "run_id": f"run-{run}",
                "ts": f"2026-01-01T00:00:{i:03d}Z",
                "latitude": 40.0,
                "longitude": -77.0 + i * lon_step,
                "vision_motion": 0.5,
                "speed_mps": 12.0,
            })
    return pd.DataFrame(rows)


def _minimum_haversine_m(a: pd.DataFrame, b: pd.DataFrame) -> float:
    import numpy as np
    from sklearn.neighbors import BallTree
    radius=6_371_000.0
    tree=BallTree(np.radians(b[["latitude","longitude"]].to_numpy(float)),metric="haversine")
    dist,_=tree.query(np.radians(a[["latitude","longitude"]].to_numpy(float)),k=1)
    return float(dist[:,0].min()*radius)


def test_spatial_split_enforces_metric_purge_margin():
    parts=spatial_train_val_test_split(
        _continuous_route_table(), block_size_m=300.0, val_fraction=0.2,
        test_fraction=0.2, seed=5, purge_margin_m=35.0,
    )
    heldout=pd.concat([parts.validation,parts.test],ignore_index=True)
    assert _minimum_haversine_m(parts.train,heldout) >= 35.0 - 1e-6
    assert _minimum_haversine_m(parts.validation,parts.test) >= 35.0 - 1e-6
    assert parts.purge_margin_m == 35.0
    assert parts.purged_train_rows + parts.purged_validation_rows > 0


def test_spatial_purge_resegments_removed_gaps():
    parts=spatial_train_val_test_split(
        _continuous_route_table(n_points=300), block_size_m=250.0,
        val_fraction=0.2, test_fraction=0.2, seed=3, purge_margin_m=50.0,
    )
    for partition in (parts.train, parts.validation):
        for _, group in partition.groupby("sequence_group_id"):
            positions=group["_source_pos"].sort_values().to_numpy()
            if len(positions)>1:
                assert (positions[1:] - positions[:-1] == 1).all()


def test_spatial_split_rejects_negative_purge_margin():
    with pytest.raises(ValueError):
        spatial_train_val_test_split(_spatial_table(), purge_margin_m=-1.0)


def test_split_boundary_temporal_features_cannot_depend_on_removed_or_other_partition_rows():
    parts = spatial_train_val_test_split(
        _continuous_route_table(n_points=300), block_size_m=250.0,
        val_fraction=0.2, test_fraction=0.2, seed=3, purge_margin_m=50.0,
    )
    for partition in (parts.train, parts.validation, parts.test):
        first = partition.groupby("sequence_group_id", sort=False).head(1)
        assert (first["vision_motion"] == 0.0).all()
        assert (first["speed_mps"] == 0.0).all()
        assert first["temporal_boundary_reset"].all()
        nonfirst = partition.loc[~partition.index.isin(first.index)]
        if not nonfirst.empty:
            assert (~nonfirst["temporal_boundary_reset"]).all()


def test_boundary_reset_is_noop_for_other_per_sample_features():
    frame = pd.DataFrame({
        "sequence_group_id": ["a", "a", "b"],
        "vision_motion": [9.0, 8.0, 7.0],
        "speed_mps": [6.0, 5.0, 4.0],
        "vibration_rms": [1.0, 2.0, 3.0],
    })
    out = reset_partition_boundary_temporal_features(frame)
    assert out["vision_motion"].tolist() == [0.0, 8.0, 0.0]
    assert out["speed_mps"].tolist() == [0.0, 5.0, 0.0]
    assert out["vibration_rms"].tolist() == [1.0, 2.0, 3.0]


def test_reloaded_spatial_test_selection_reapplies_boundary_resets():
    df = _continuous_route_table(n_points=300)
    parts = spatial_train_val_test_split(
        df, block_size_m=250.0, val_fraction=0.2, test_fraction=0.2, seed=3, purge_margin_m=50.0,
    )
    reloaded = select_spatial_blocks(df, parts.test_blocks, block_size_m=250.0)
    first = reloaded.groupby("sequence_group_id", sort=False).head(1)
    assert not first.empty
    assert (first["vision_motion"] == 0.0).all()
    assert (first["speed_mps"] == 0.0).all()
