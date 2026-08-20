# ML and multimodal modeling

## Tasks

RailGuard-MM uses multi-task temporal learning rather than a single classification head:

1. forecast future vibration RMS;
2. forecast future visual-motion score;
3. estimate anomaly probability when legitimate labels exist;
4. learn a fused temporal representation for diagnostics.

## Deployment feature contract

The native deployment interface is intentionally explicit and ordered:

```text
[vibration_rms,
 vibration_peak,
 vibration_kurtosis,
 crest_factor,
 vision_motion,
 vision_contrast,
 speed_mps,
 temperature_c,
 humidity]
```

`ml/railguard_ml/contracts.py` is the Python source of truth and `edge/cpp/include/railguard/model_contract.hpp` mirrors it for native deployment. ONNX export refuses a checkpoint whose feature names/order do not match this contract, and the TensorRT backend verifies the final tensor dimension before inference. This prevents a retrained model from silently consuming semantically reordered inputs.

The public Rail-VIVID adapter also preserves six individual accelerometer RMS channels for experiments, but they are not silently inserted into the deployment contract because the public six-sensor arrangement is not assumed to map one-to-one onto the proposed three triaxial IIS3DWB nodes.

## Windowing

- modeling rate: 10 Hz / 100 ms;
- sequence length: 32 steps (3.2 s);
- prediction horizons: +1, +5 and +10 steps;
- native ~19 ms IIS3DWB feature windows are resampled onto this fixed timebase before inference;
- camera frames are nearest-neighbor associated in the host monotonic clock domain after GNSS/PPS clock alignment;
- speed/environment are continuous context variables;
- latitude/longitude are excluded from the default model inputs.

## Classical baseline

`ml/train_classical.py` uses the same structured feature contract as the deep model and trains a `RandomForestRegressor` for +1/+5/+10 vibration forecasts. Persistence is reported on the same untouched test partition. `IsolationForest` remains the unsupervised anomaly baseline.

## Deep model

`FusionTransformer` has three stages:

### Vision encoder

A compact CNN maps each frame to an embedding. The deployment camera is monochrome Y12, so public training images are converted to luminance and replicated across three channels (`monochrome_replicated_rgb`) before entering the CNN. This preserves the export-friendly three-channel tensor shape without teaching the model color cues that the deployed sensor cannot observe. The image mode is a versioned checkpoint/export/runtime contract rather than an undocumented preprocessing choice.

### Sensor encoder

The nine structured features are standardized using **training-split statistics only**, then projected into the same latent dimension as vision.

### Temporal fusion

The vision and structured embeddings are fused per time step. A Transformer encoder processes the synchronized sequence, and the final hidden state drives +1/+5/+10 vibration forecasts, +1/+5/+10 vision forecasts and an anomaly logit. The vibration head uses a positive transform and the visual-motion head is bounded to `[0,1]`, matching physical deployment semantics rather than relying on downstream clipping. Training also fails fast on non-finite required inputs instead of allowing NaN context to flow through normalization.


## Auxiliary railway-surface visual pretraining

`ml/train_visual_faults.py` trains the same compact `FrameEncoder` on the openly licensed Railway Track Surface Faults Dataset (CC BY 4.0, DOI `10.17632/8hxtgyyxrw.2`). The task has seven surface-fault classes and uses the deployed `monochrome_replicated_rgb` image contract.

The auxiliary classification score is deliberately **not** the project's headline machine-vision claim because the source images were extracted from inspection video and neighboring frames may be correlated. The value experiment is transfer to the primary Rail-VIVID benchmark:

1. train/fingerprint the auxiliary encoder;
2. initialize `vision_only` and `multimodal` Rail-VIVID models with `--frame-encoder-init`;
3. keep the sensor-only baseline unchanged;
4. compare the final untouched-spatial-test result with an otherwise identical random-initialization experiment.

Visual-pretraining source DOI, license, image contract and dataset SHA-256 are stored in the Rail-VIVID checkpoint. Independent model comparison requires the `vision_only` and `multimodal` checkpoints to use the same auxiliary pretraining fingerprint.

## Anomaly supervision

No anomaly catalog means no supervised anomaly label. Preprocessing stores `NaN`; training applies BCE only when labeled training windows contain both classes, and AUROC/AP are reported only when the untouched test partition supports them. Forecast residuals and Isolation Forest remain valid unsupervised signals independently of that head.

## Leakage controls

The **default** benchmark is a spatially disjoint split:

- GPS positions are assigned to 500 m geographic blocks;
- every block belongs to exactly one of train/validation/test across all repeated passes;
- a default 30 m haversine purge margin removes train rows near validation/test geography and validation rows near test geography;
- the measured minimum train↔held-out and validation↔test separations are saved with the checkpoint;
- sequence IDs change at block boundaries and are rebuilt after purging; split-sensitive temporal features are reset at those final boundaries so their previous-row computation cannot cross partitions;
- forecast targets are generated inside the final contiguous sequence group;
- scalers use the training partition only;
- validation selects checkpoints using the versioned joint forecast objective **vibration Huber + 0.4 × visual-motion Huber**;
- checkpoint metadata records the training-protocol version, image mode and selection objective;
- the test partition remains untouched until final evaluation.

A run-disjoint split remains available as a secondary experiment, but it is weaker on a repeated route because the model could otherwise see the same geographic scene during another pass.

## Untouched-test evaluation

`ml/evaluate_multimodal.py` reports:

- persistence;
- nonvisual dependency ablation (vibration + speed/environment only);
- visual dependency ablation (camera + engineered visual scalars);
- full multimodal fusion;
- horizon-specific MAE/RMSE;
- anomaly AUROC/AP when supported by labels.

The two dependency ablations mask inputs of **one trained fusion model**. They are not mislabeled as independently trained unimodal networks.

For the stronger modality experiment, `ml/train_multimodal.py --modality {sensor_only,vision_only,multimodal}` trains three separate models. `ml/evaluate_independent_models.py` refuses to compare them unless dataset fingerprint, architecture/training protocol, feature/image contract, sequence length, timebase, split geography, purge margin **and training seed** match exactly. This separates the question “what did one fusion checkpoint depend on?” from the stronger question “does an independently optimized multimodal model outperform independently optimized unimodal models?”

The independent evaluator also reports paired uncertainty by resampling whole held-out geographic blocks/runs rather than autocorrelated windows. With at least three held-out groups it publishes a paired bootstrap interval and probability of improvement; with fewer groups it deliberately withholds a nominal 95% interval. `ml/summarize_seed_runs.py` then aggregates several matched-seed experiments and reports between-seed mean/SD plus win–tie–loss counts. Seed variability is kept distinct from geographic generalization uncertainty.

### Synchronization robustness

The evaluator also runs inference-time robustness tests on the untouched test split:

- fixed camera offsets (default +1 and +2 model steps) without circular wrap;
- camera frame dropout (default 10% and 30%) using hold-last-frame behavior.

These quantify whether performance collapses under realistic temporal association failures and are particularly relevant to Rail-VIVID's documented vision timestamp drift.


## Portfolio experiment orchestration

`ml/run_showcase_experiments.py` is the single entry point for the result that belongs on the GitHub front page. It runs the classical baseline, matched-seed independently trained modality models, untouched-test evaluation, held-out-group uncertainty, timing/dropout robustness and repeated-seed aggregation. `scripts/render_showcase_results.py` converts the generated artifacts into a concise README-ready result card. See [experiments.md](experiments.md) for the exact evidence hierarchy and publication gates.

## Deployment optimization

ONNX tensors are:

- `frames`: `[1,T,3,96,96]` FP32 in `[0,1]`;
- `sensors`: `[1,T,9]` physical-unit features in the explicit contract above;
- `vibration`: `[1,3]`;
- `vision`: `[1,3]`;
- `anomaly_probability`: `[1]`.

The ONNX wrapper embeds training normalization. Export writes a deployment lineage manifest binding the processed-data/checkpoint provenance to the ONNX SHA-256; the TensorRT builder adds the engine SHA-256. `deploy/tensorrt/run_edge_verified.sh` refuses to launch an engine whose hash does not match that manifest and derives sequence length/model timebase from the verified manifest, so operators cannot silently run a valid engine under the wrong temporal contract. TensorRT uses a dynamic temporal profile from `T=4` to `T=128`, optimized around the checkpoint sequence length. FP16 is the first deployment target; INT8 should only be accepted after representative calibration and untouched-test accuracy comparison.

## Evidence still required for a final ML claim

The repository implements the evaluation path but does not invent accuracy. A publishable portfolio result should include the generated untouched-spatial-test table, robustness sensitivity tables and the exact checkpoint/split metadata used to produce them.

## Reproducibility and provenance

Deep training accepts `--seed` and an optional `--deterministic` mode. Saved checkpoints bind the **multimodal dataset**, not only the CSV: the provenance fingerprint canonicalizes the structured rows after replacing each image path with the SHA-256 of the referenced image bytes. Evaluation refuses a checkpoint if either structured values or referenced image content differs. Checkpoints additionally record the raw processed-table SHA-256, training seed, Python/PyTorch/NumPy/pandas versions, platform and Git commit when available. The source downloader separately records the immutable Hugging Face revision and per-file hashes. Classical reports carry the structured-input provenance appropriate to their non-image baseline. This does not imply bit-identical results across every GPU/driver combination; it makes the experimental input and software context auditable.

## Between-location generalization

The untouched-test evaluator reports global metrics **and** per-held-out-group vibration MAE. For spatial evaluation, each group is a geographic block; for the secondary run split, each group is a physical run. The report includes median/IQR/min/max across groups so a low global average cannot hide a small number of badly generalized locations.
