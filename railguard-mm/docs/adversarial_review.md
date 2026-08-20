# Adversarial engineering review

This document is intentionally written from the perspective of a skeptical senior reviewer trying to invalidate RailGuard-MM's strongest claims. It distinguishes weaknesses that were fixed in code from evidence that still requires public-data training or physical target hardware.

## High-severity findings closed in the repository

| Attack | Why it mattered | Current control |
|---|---|---|
| "Your test set was used for checkpoint selection." | Validation performance is not untouched-test evidence. | Validation chooses checkpoints; a separate test group list is stored and opened only by the final evaluator. |
| "Repeated train passes show the same track as test." | Run-disjoint splitting still permits scene/location memorization. | Default 500 m spatial blocks across all passes plus a 30 m row-level geodesic purge margin. |
| "A block boundary can put train and test centimeters apart." | Grid IDs alone do not create a physical exclusion zone. | Haversine nearest-neighbor purging; measured minimum separations are checkpoint metadata. |
| "Your classical future label can point into a purged region." | Target construction before purging leaks held-out geography. | Classical targets are generated after final split/purge inside re-segmented groups. |
| "Filtering can stitch a Transformer window across a removed gap." | Resetting DataFrame indices can manufacture false temporal continuity. | `_source_pos` continuity is preserved and sequence IDs are rebuilt after purging; dataset construction also detects index gaps. |
| "Your no-vision ablation still contains visual features." | Engineered motion/contrast leaked vision into the ablation. | Camera embeddings and visual structured columns are masked together. |
| "Unlabeled track is being trained as normal." | All-zero anomaly labels manufacture supervision. | No catalog => `NaN`; supervised anomaly loss/metrics are gated on real labels. |
| "Camera and MCU timestamps are different clocks." | Nearest timestamps are meaningless without a clock-domain transform. | GNSS/PPS UTC↔Linux-monotonic aligner with lock/jitter telemetry. |
| "Training is 10 Hz but the MCU emits ~52 Hz windows." | Sequence dynamics differ between training and deployment. | Fixed-rate 100 ms spatial resampler before temporal inference. |
| "The selected monochrome camera does not output YUYV." | A hard-coded color path would fail on the chosen hardware. | V4L2 negotiates Y12 first and host-tested preprocessing supports Y12/GREY/YUYV with stride. |
| "Bad-but-CRC-valid floats can poison the model/cloud." | CRC checks transport, not semantics. | C++ and Python semantic validation reject non-finite/implausible values; missing context is `null` and inference fails closed. |
| "Network QoS can eventually block acquisition." | Synchronous broker acknowledgements can backpressure stdout. | SQLite durable producer/consumer spool with independent MQTT drain, size bound and drop counter. |
| "Any TensorRT engine can publish as the same model." | Results are not auditable without artifact identity. | checkpoint→ONNX→engine SHA-256 lineage manifest; verified launcher binds `model_version`. |
| "Your native fallback publishes fake ML." | A heuristic diagnostic can be mistaken for trained inference. | No engine => inference disabled; diagnostic reference engine is self-test only. |

## Remaining attacks that still succeed

### 1. There is no published Rail-VIVID model-quality result yet

The experiment machinery is stronger than the evidence. The repository can produce persistence, dependency-ablation, per-location and temporal-robustness metrics, but it does not invent an accuracy table without training on the public dataset. The flagship release should add the untouched-spatial-test JSON/Markdown artifacts and plots from an exact checkpoint manifest.

### 2. TensorRT performance is not measured on the Jetson

The CPU native microbenchmarks are real; Jetson FP32/FP16/INT8 latency, power, memory and accuracy deltas still require the target NVIDIA platform. Treat build/benchmark scripts as an evaluation protocol, not as performance evidence.

### 3. The selected camera is not yet a zero-copy GPU pipeline

The Y12 path is now correct, but it still converts/resizes on CPU before host→device inference transfer. A later Jetson iteration should evaluate V4L2/GStreamer DMA-BUF/NVMM import and CUDA preprocessing, then quantify whether complexity is justified by end-to-end p95 latency/power.

### 4. Software frame timestamps are not yet a measured exposure edge

The runtime correctly aligns GNSS/PPS time to the V4L2 monotonic clock and supports the selected Y12 camera format, but the kernel frame timestamp is still being treated as the camera observation time. The camera exposes 1.8 V TRIG/STROBE GPIO; a later hardware revision should level-translate STROBE into an STM32 timer capture and compare that physical exposure event with the V4L2 timestamp distribution before tightening synchronization thresholds. Until then, the current synchronization metric is software-frame timing, not oscilloscope-proven exposure timing.

### 5. Public-data and reference-hardware vibration topology differ

Rail-VIVID publishes six scalar accelerometer channels; the reference node uses three triaxial IIS3DWB devices. The adapter preserves all six public per-channel RMS values but the deployable model uses topology-agnostic fused vibration statistics. A physical campaign should retrain/fine-tune on the actual three-location hardware before claiming transfer equivalence.

### 6. The acquisition electronics are not physically validated

The repo contains a concrete reference design, BOM, netlist, pin map, firmware boundary, FMEA and bring-up plan. It does not contain oscilloscope/logic-analyzer traces, a manufactured PCB, environmental/EMC results or a sustained FIFO-overrun campaign. Those are intentionally hardware-required evidence.

### 7. Frontend dependency resolution is not locally reproduced here

GitHub CI includes the Node build, but the current local execution environment could not reach the npm registry to generate a lockfile. A public release should commit a package lock generated in a normal network environment and use `npm ci` in CI/container builds.

## Release gate

A strong public tag should require all locally provable items below:

```bash
./scripts/validate_all.sh
make benchmark
python scripts/sync_evidence.py --write
python scripts/sync_evidence.py --check
```

External evidence is a separate gate: untouched Rail-VIVID metrics, Jetson TensorRT measurements and physical acquisition-board validation must be labeled with the exact hardware/dataset/model provenance that produced them.
