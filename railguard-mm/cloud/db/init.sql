CREATE EXTENSION IF NOT EXISTS timescaledb;

CREATE TABLE IF NOT EXISTS telemetry (
  ts TIMESTAMPTZ NOT NULL,
  device_id TEXT NOT NULL,
  seq BIGINT NOT NULL,
  schema_version SMALLINT NOT NULL DEFAULT 1,
  sample_period_ms DOUBLE PRECISION NOT NULL DEFAULT 100.0,
  lat DOUBLE PRECISION,
  lon DOUBLE PRECISION,
  speed_mps DOUBLE PRECISION,
  temperature_c DOUBLE PRECISION,
  humidity DOUBLE PRECISION,
  vibration_rms DOUBLE PRECISION,
  vibration_peak DOUBLE PRECISION,
  vibration_kurtosis DOUBLE PRECISION,
  crest_factor DOUBLE PRECISION,
  band0 DOUBLE PRECISION,
  band1 DOUBLE PRECISION,
  band2 DOUBLE PRECISION,
  band3 DOUBLE PRECISION,
  vision_motion DOUBLE PRECISION,
  vision_contrast DOUBLE PRECISION,
  vision_sharpness DOUBLE PRECISION,
  packet_loss BIGINT,
  spool_depth BIGINT,
  spool_dropped BIGINT,
  camera_matched BOOLEAN,
  sync_error_ms DOUBLE PRECISION,
  sensor_skew_ms DOUBLE PRECISION,
  clock_alignment_locked BOOLEAN,
  clock_jitter_ms DOUBLE PRECISION,
  clock_samples BIGINT,
  context_flags SMALLINT,
  sensor0_rms DOUBLE PRECISION,
  sensor1_rms DOUBLE PRECISION,
  sensor2_rms DOUBLE PRECISION,
  raw JSONB,
  PRIMARY KEY (device_id, ts, seq)
);
SELECT create_hypertable('telemetry', by_range('ts'), if_not_exists => TRUE);
-- Idempotent upgrades for an existing development volume.
ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS schema_version SMALLINT NOT NULL DEFAULT 1;
ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS spool_dropped BIGINT;
ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS vision_sharpness DOUBLE PRECISION;
ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS camera_matched BOOLEAN;
ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS sync_error_ms DOUBLE PRECISION;
ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS sensor_skew_ms DOUBLE PRECISION;
ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS sensor0_rms DOUBLE PRECISION;
ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS sensor1_rms DOUBLE PRECISION;
ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS sensor2_rms DOUBLE PRECISION;
ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS sample_period_ms DOUBLE PRECISION NOT NULL DEFAULT 100.0;
ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS clock_alignment_locked BOOLEAN;
ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS clock_jitter_ms DOUBLE PRECISION;
ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS clock_samples BIGINT;
ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS context_flags SMALLINT;

CREATE TABLE IF NOT EXISTS predictions (
  issued_at TIMESTAMPTZ NOT NULL,
  target_ts TIMESTAMPTZ NOT NULL,
  device_id TEXT NOT NULL,
  source_seq BIGINT NOT NULL,
  model_version TEXT NOT NULL,
  horizon_steps INTEGER NOT NULL,
  step_ms DOUBLE PRECISION NOT NULL DEFAULT 100.0,
  vibration_rms_pred DOUBLE PRECISION,
  vision_motion_pred DOUBLE PRECISION,
  anomaly_probability DOUBLE PRECISION,
  PRIMARY KEY (device_id, issued_at, source_seq, target_ts, model_version)
);
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS step_ms DOUBLE PRECISION NOT NULL DEFAULT 100.0;
ALTER TABLE predictions ADD COLUMN IF NOT EXISTS source_seq BIGINT;
SELECT create_hypertable('predictions', by_range('target_ts'), if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS telemetry_device_ts_idx ON telemetry(device_id, ts DESC);
CREATE INDEX IF NOT EXISTS predictions_device_target_idx ON predictions(device_id, target_ts DESC);
