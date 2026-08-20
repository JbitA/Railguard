import pandas as pd

from ml.railguard_ml.dataset import RailSequenceDataset


def _rows(count: int, group: str = "g"):
    return [{
        "run_id": "run-0",
        "sequence_group_id": group,
        "image_path": "unused.jpg",
        "vibration_rms": float(i),
        "vision_motion": float(i),
        "feature": float(i),
    } for i in range(count)]


def test_exact_minimum_forecast_span_yields_one_window():
    # seq_len + maximum horizon = 32 + 10 rows is exactly one valid window.
    df=pd.DataFrame(_rows(42))
    ds=RailSequenceDataset(df,["feature"],seq_len=32)
    assert len(ds)==1
    assert ds.starts == [0]


def test_noncontiguous_group_indices_are_not_stitched():
    rows=_rows(25,"a") + _rows(1,"b") + _rows(25,"a")
    df=pd.DataFrame(rows)
    ds=RailSequenceDataset(df,["feature"],seq_len=20)
    # Each A span is only 25 rows (<20+10) and must not be joined across B.
    assert len(ds)==0
