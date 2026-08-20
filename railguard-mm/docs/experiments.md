# Experiment program: proving the value of RailGuard-MM

The portfolio should not ask a reviewer to infer value from architecture. It should answer a short set of falsifiable questions with reproducible evidence.

## Headline questions

1. **Does multimodal fusion improve future condition estimates?**
2. **Does the improvement survive unseen track geography and repeated training seeds?**
3. **How sensitive is fusion to camera timing error and dropped frames?**
4. **Does rail-specific visual pretraining improve the deployed vision encoder?**
5. **Does the native edge implementation materially reduce latency while preserving the same data contract?**

The first two questions are the scientific headline. The remaining experiments explain *why the system is credible and deployable*.

---

## Experiment E0 — public-data quality gate

**Dataset:** Rail-VIVID (CC BY 4.0).

Before training, every processed run emits QA for:

- accepted/rejected 100 ms windows;
- p50/p95/max image↔telemetry timestamp error;
- missing/non-finite model inputs;
- geographic coverage and repeated-run identity;
- dataset and referenced-image SHA-256 provenance.

**Publish:** a compact table of run count, modeling windows, accepted frame rate and p95 alignment error. Do not publish model metrics until this gate is clean.

---

## Experiment E1 — does multimodal fusion add predictive value?

### Models

Train separately, with matched split and training seed:

- persistence baseline;
- Random Forest structured baseline;
- `sensor_only` temporal model;
- `vision_only` temporal model;
- `multimodal` fusion model.

The neural models are *independently trained*. This is intentionally stronger than masking modalities after a fusion model has already learned.

### Split

Primary evaluation uses:

- 500 m geographic blocks;
- 30 m geodesic purge margin by default;
- no temporal window crossing a purged boundary;
- no latitude/longitude in the learned deployment feature vector;
- untouched test geography used only after checkpoint selection.

### Metrics

Report, for +1/+5/+10 model steps:

- vibration MAE and RMSE;
- visual-motion MAE and RMSE;
- persistence-relative improvement;
- per-held-out-geographic-block metrics;
- paired held-out-group bootstrap confidence interval for multimodal error reduction;
- matched-seed mean/SD and win/tie/loss counts.

### Headline evidence table

Populate this table only from generated experiment artifacts:

| Model | Vib MAE +1 | Vib MAE +5 | Vib MAE +10 | Vision MAE +1 | +5 | +10 |
|---|---:|---:|---:|---:|---:|---:|
| Persistence | — | — | — | — | — | — |
| Random Forest | — | — | — | n/a | n/a | n/a |
| Sensor-only DL | — | — | — | — | — | — |
| Vision-only DL | — | — | — | — | — | — |
| **Multimodal DL** | **—** | **—** | **—** | **—** | **—** | **—** |

A result is README-headline quality only when the same untouched-test contract is used across models and at least three matched neural training seeds are available.

---

## Experiment E2 — synchronization and camera-loss robustness

A multimodal model is only useful if it degrades predictably when modalities lose alignment.

For the selected multimodal checkpoint, evaluate:

- fixed camera offsets of ±1 and ±2 model steps;
- deterministic frame dropout at 10% and 30%;
- held-last-frame behavior without circular wrapping or future leakage.

Report the relative degradation from the aligned/no-dropout result.

**Value:** this connects the ML result directly to the system-design work on PPS, clock alignment, nearest-frame association and camera health telemetry.

---

## Experiment E3 — does railway-specific visual pretraining help?

**Auxiliary source:** Railway Track Surface Faults Dataset, DOI `10.17632/8hxtgyyxrw.2`, CC BY 4.0.

The dataset contains seven fault classes: Cracks, Flakings, Grooves, Joints, Shellings, Spallings and Squats. It is used only as an auxiliary vision task; it is never treated as Rail-VIVID time-series ground truth.

### Procedure

1. Train the same compact `FrameEncoder` used by RailGuard on the seven-class auxiliary task.
2. Convert all images to the deployed `monochrome_replicated_rgb` contract.
3. Save only a provenance-bound encoder checkpoint.
4. Train matched `vision_only` and `multimodal` Rail-VIVID models with and without that initialization.
5. Compare **Rail-VIVID untouched-spatial-test** performance, not merely auxiliary classification accuracy.

The auxiliary dataset consists of frames extracted from inspection video, so a simple image split may contain correlated neighboring frames. Its classification score is therefore secondary evidence. The real portfolio question is whether the initialization improves generalization on Rail-VIVID.

Commands:

```bash
python ml/open_data.py import-surface-faults /path/to/provider/download \
  --dest data/rail_surface_faults

python ml/train_visual_faults.py /path/to/provider/download \
  --out models/rail_surface_frame_encoder.pt

python ml/run_showcase_experiments.py data/processed/rail_vivid_multirun.csv \
  --seeds 7,17,37 \
  --visual-init models/rail_surface_frame_encoder.pt
```

---

## Experiment E4 — native edge value

The measured host benchmark compares the Python reference with the C++20 implementation for:

- production packet decoding;
- semantic/timestamp validation;
- validated packet throughput;
- matched 512-sample vibration DSP.

These are CPU microbenchmarks and are already measured/reproducible. Jetson TensorRT numbers remain a target-hardware experiment.

For the Jetson campaign report:

- PyTorch CUDA;
- ONNX Runtime;
- TensorRT FP32;
- TensorRT FP16;
- TensorRT INT8;
- p50/p95 inference and end-to-end latency;
- memory and power;
- accuracy delta after quantization.

---

## Experiment E5 — edge/cloud failure semantics

The portfolio claim is not simply "uses MQTT". It is that an observation survives transient network/cloud failure without confusing broker acceptance with database persistence.

The integration test demonstrates:

1. edge record committed to SQLite;
2. MQTT QoS 1 publish;
3. TimescaleDB telemetry + prediction persistence;
4. application ACK only after persistence;
5. exact `(device_id, timestamp, sequence)` ACK identity;
6. HMAC verification when configured;
7. outbox deletion only after verified application ACK.

The GitHub CI integration job runs this composed path when Docker is available.

---

# Reproducible showcase suite

After preparing Rail-VIVID, run the complete experiment plan with:

```bash
python ml/run_showcase_experiments.py \
  data/processed/rail_vivid_multirun.csv \
  --seeds 7,17,37 \
  --epochs 10 \
  --out artifacts/showcase
```

This produces a directory organized by seed plus:

```text
artifacts/showcase/
├── experiment_manifest.json
├── classical/
│   └── metrics.json
├── seed_7/
│   ├── sensor_only.pt
│   ├── vision_only.pt
│   ├── multimodal.pt
│   ├── independent_metrics.json
│   ├── independent_metrics.md
│   ├── robustness.json
│   └── robustness.md
├── seed_17/
├── seed_37/
├── seed_summary.json
└── seed_summary.md
```

Every neural comparison is refused if dataset fingerprint, feature contract, image contract, split, purge margin, untouched geography, timebase, checkpoint objective or training seed do not match.

## What belongs on the front page

Promote only these categories to the README:

- measured native benchmark results;
- real public-data QA statistics;
- real untouched-test E1 results;
- real E2 robustness degradation;
- real E3 pretraining effect;
- actual Jetson/hardware measurements when available.

Synthetic dashboard/demo data is useful for showing the UI and integration path, but must remain visually labeled as **synthetic integration demo — not model-accuracy evidence**.
