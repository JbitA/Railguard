from datetime import datetime, timezone

import numpy as np

from cloud.ml_worker.worker import _trend_forecast, infer_one


def test_trend_forecast_constant_series_is_persistence():
    assert _trend_forecast(np.ones(12) * 3.25, (1, 5, 10)) == [3.25, 3.25, 3.25]


class _FakeResult:
    def __init__(self, rows): self._rows=rows
    def fetchall(self): return self._rows


class _FakeConn:
    def __init__(self, rows):
        self.rows=rows
        self.inserts=[]
    def execute(self, sql, args=None):
        if "SELECT ts, seq, sample_period_ms" in sql:
            return _FakeResult(self.rows)
        if "INSERT INTO predictions" in sql:
            self.inserts.append(args)
            return _FakeResult([])
        raise AssertionError(sql)


def test_cloud_baseline_uses_recorded_model_timebase():
    t=datetime(2026,1,1,tzinfo=timezone.utc)
    rows=[{
        "ts":t,
        "seq":i,
        "sample_period_ms":200.0,
        "vibration_rms":float(i+1),
        "vision_motion":float(i)/10,
        "speed_mps":1.0,
    } for i in reversed(range(8))]
    conn=_FakeConn(rows)
    infer_one(conn,"railguard-01")
    assert len(conn.inserts)==3
    assert [r[5] for r in conn.inserts] == [1,5,10]
    assert all(r[6] == 200.0 for r in conn.inserts)
    assert [int((r[1]-t).total_seconds()*1000) for r in conn.inserts] == [200,1000,2000]
    assert all(r[3] == 7 for r in conn.inserts)
    assert all(r[4] == "cloud-trend-baseline-v2" for r in conn.inserts)


def test_trend_forecast_can_enforce_physical_output_bounds():
    falling = np.array([1.0, 0.5, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert all(x >= 0.0 for x in _trend_forecast(falling, (1, 5, 10), lower=0.0))
    rising = np.linspace(0.7, 1.0, 16)
    bounded = _trend_forecast(rising, (1, 5, 10), lower=0.0, upper=1.0)
    assert all(0.0 <= x <= 1.0 for x in bounded)
