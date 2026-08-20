from cloud.ingestor.transform import flatten


def _record():
    return {
      "schema_version":1,"device_id":"d","ts":"2026-01-01T00:00:00Z","seq":4,"sample_period_ms":100.0,
      "gps":{"lat":1.0,"lon":2.0,"speed_mps":3.0},"environment":{"temperature_c":20.0,"humidity":0.5},
      "vibration":{"rms_ms2":2.0,"peak_ms2":5.0,"kurtosis":3.0,"crest_factor":2.5,"band_energy":[1,2,3,4],
                   "sensors":[{"sensor_id":0,"rms_ms2":1.1},{"sensor_id":1,"rms_ms2":2.2},{"sensor_id":2,"rms_ms2":3.3}]},
      "vision":{"motion_score":0.2,"contrast":0.7,"sharpness":1.2},
      "health":{"packet_loss":1,"spool_depth":2,"camera_matched":True,"sync_error_ms":-4.5,"sensor_skew_ms":8.0,
                "clock_alignment_locked":True,"clock_jitter_ms":2.0,"clock_samples":32,"context_flags":3}
    }


def test_flatten_preserves_sync_spatial_clock_and_timebase_metrics():
    row=flatten(_record())
    assert row['camera_matched'] is True and row['sync_error_ms']==-4.5 and row['sensor_skew_ms']==8.0
    assert (row['sensor0_rms'],row['sensor1_rms'],row['sensor2_rms'])==(1.1,2.2,3.3)
    assert row['sample_period_ms']==100.0
    assert row['clock_alignment_locked'] is True and row['clock_jitter_ms']==2.0 and row['clock_samples']==32
    assert row['context_flags']==3
