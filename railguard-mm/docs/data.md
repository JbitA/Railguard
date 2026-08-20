# Open-data strategy

RailGuard-MM separates datasets by the claim they are allowed to support. This avoids the common portfolio problem of mixing unrelated public datasets until it becomes unclear what a result means.

The machine-readable registry is [`data/open_sources.yaml`](../data/open_sources.yaml). List it with:

```bash
python ml/open_data.py list
```

## Source 1 — Rail-VIVID: primary multimodal benchmark

- **License:** CC BY 4.0
- **Dataset DOI:** `10.57967/hf/8411`
- **Paper DOI:** `10.1038/s41597-026-07555-y`
- **Repository:** `saluslab/Rail-VIVID`
- **Role:** headline multimodal/time-series experiment

Rail-VIVID is the primary real-world source because it combines six vibration channels, global-shutter vision, GPS and environmental context over repeated railway runs. Its dataset card also documents nine ground-truth track anomalies.

The repository treats Rail-VIVID and the proposed three-IIS3DWB acquisition node as **related but not sensor-identical systems**. Training therefore uses cross-platform aggregate vibration statistics in the deployment feature contract rather than pretending the six public accelerometers map one-to-one onto the three proposed triaxial sensors. Per-channel RMS values remain available for research.

### Reproducible acquisition

The unified acquisition entry point resolves mutable Hugging Face revisions to an immutable commit and hashes every selected file:

```bash
python ml/open_data.py acquire rail_vivid \
  --pattern 'AtoB_*/*' \
  --dest data/rail_vivid
```

For finer control the original downloader remains available:

```bash
python ml/download_rail_vivid.py --list
python ml/download_rail_vivid.py \
  --pattern 'AtoB_*/*' \
  --dest data/rail_vivid
```

Both paths write a manifest recording repository commit, file size and SHA-256. Training provenance later fingerprints both the processed table and every referenced training image.

### Dataset-specific cautions handled by the adapter

1. **Temperature units** — `Temperature` is published in °F; preprocessing converts it to °C.
2. **Time zones** — CSV timestamps are wall-clock strings while image filenames carry Unix-like timestamps. The adapter localizes to `America/New_York` by default and converts to UTC before frame matching.
3. **Vision alignment uncertainty** — the dataset authors document variable-FPS timestamp drift that can displace image location by up to roughly 15 m. Every processed run gets an alignment QA JSON with accepted/rejected windows and p50/p95/max nearest-frame timestamp error.
4. **Explicit frame rejection** — nearest images farther than `--max-image-delta` (default 250 ms) are rejected instead of silently paired.
5. **Leakage** — the headline benchmark uses geographically disjoint blocks across repeated runs. Default: 500 m blocks plus a 30 m geodesic purge margin.
6. **Sequence boundaries** — temporal windows and forecast targets cannot cross run/spatial-block boundaries or geographic purge gaps.
7. **Location memorization** — latitude/longitude are used for splitting/visualization/anomaly association but excluded from the learned deployment feature vector.
8. **Anomaly supervision** — anomaly BCE/AUROC is enabled only when an explicit ground-truth catalog has been associated. Unknown is represented as `NaN`, not silently as normal.

## Source 2 — Railway Track Surface Faults: auxiliary vision benchmark

- **License:** CC BY 4.0
- **DOI:** `10.17632/8hxtgyyxrw.2`
- **Provider:** Mendeley Data
- **Classes:** Cracks, Flakings, Grooves, Joints, Shellings, Spallings, Squats
- **Role:** auxiliary rail-surface representation/pretraining task

This dataset was collected from cameras mounted on a railway inspection vehicle and manually labeled by surface-fault class. It is useful for giving the compact frame encoder an explicit railway-surface semantics task.

It is **not** merged with Rail-VIVID labels and its classification score is **not** presented as the project's generalization headline. Frames originated from video and can be temporally correlated, so the high-value experiment is whether this pretraining improves untouched-spatial-test performance on Rail-VIVID.

### Acquisition and provenance

Download Version 2 from the provider landing page:

`https://data.mendeley.com/datasets/8hxtgyyxrw/2`

Then verify the seven expected class folders and fingerprint the exact image bytes:

```bash
python ml/open_data.py import-surface-faults /path/to/provider/download \
  --dest data/rail_surface_faults
```

Use `--copy` only if you want a normalized local copy. By default the tool writes a content manifest/index without duplicating the provider download.

Train the auxiliary encoder:

```bash
python ml/train_visual_faults.py /path/to/provider/download \
  --out models/rail_surface_frame_encoder.pt \
  --metrics artifacts/evaluation/visual_faults.json
```

The training path converts source images to the same `monochrome_replicated_rgb` information contract used by the deployed Y12 camera path.

## Dataset adapter

`ml/prepare_rail_vivid.py` converts one downloaded run into the common 10 Hz modeling table:

1. parse and UTC-normalize timestamps;
2. form 100 ms vibration windows across all six public channels;
3. compute aggregate RMS, peak, kurtosis and crest factor while preserving `accel_1_rms` … `accel_6_rms`;
4. find the nearest image and reject matches beyond the configured tolerance;
5. compute visual motion, contrast and sharpness/focus QA while retaining the original image path for CNN training;
6. derive speed from consecutive GPS fixes;
7. convert environmental context to repository units;
8. optionally associate a geospatial anomaly catalog;
9. emit `<run>.qa.json` with frame-alignment quality statistics.

Use `ml/combine_runs.py` to concatenate processed runs without losing run identity.

## Demo stream

`scripts/generate_demo_data.py` creates deterministic synchronized telemetry for MQTT/database/API/dashboard integration tests. It is synthetic infrastructure data and is never used as evidence of model accuracy.

## Preparing Rail-VIVID runs

```bash
python ml/prepare_rail_vivid.py data/rail_vivid/AtoB_20_1/run.csv \
  --out data/processed/AtoB_20_1.csv

python ml/combine_runs.py data/processed/*.csv \
  --out data/processed/rail_vivid_multirun.csv

python ml/run_showcase_experiments.py \
  data/processed/rail_vivid_multirun.csv \
  --seeds 7,17,37 \
  --out artifacts/showcase
```

Use the actual paths returned by `ml/download_rail_vivid.py --list`; example paths illustrate the workflow rather than depending on a particular release layout.
