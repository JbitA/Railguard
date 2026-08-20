# Verification matrix

RailGuard-MM deliberately separates **measured**, **host-verified**, **implemented**, and **hardware-required** evidence. This is the claim boundary for the repository.

| Claim | Evidence | Status |
|---|---|---|
| C17 packet framing/CRC is deterministic | `firmware/stm32_hal/host_test.c`, built with `-Wall -Wextra -Werror` | Host-verified |
| C17, C++20 and Python share the same production wire contract | Frozen **94-byte** C17 vector decoded in C++ and Python | Host-verified |
| Serial decoders recover from an incomplete false-sync header when a later CRC-valid frame is already buffered | C++/Python false-sync regression tests | Host-verified |
| CRC-valid but semantically invalid sensor values are rejected | C++ and Python tests cover NaN, invalid GNSS and invalid humidity | Host-verified |
| GNSS stream parsing survives arbitrary DMA chunk boundaries/noise | portable NMEA stream-framer host test | Host-verified |
| Vibration windows contain real statistical/spectral features | portable C17 DSP test on a 512-sample multi-axis signal | Host-verified |
| Three vibration locations stay distinct until synchronized fusion | `SpatialVibrationAggregator` requires fresh IDs 0/1/2 | CTest verified |
| Excessive inter-sensor timestamp skew is rejected | configurable `--max-sensor-skew-ms` | CTest verified |
| Camera association is nearest-timestamp, not latest-frame | bounded `FrameSynchronizer` | CTest verified |
| Selected monochrome camera's Y12 stream is represented by the native path | format-aware Y12/GREY/YUYV preprocessing with stride/padding tests | CTest verified |
| Native camera sharpness/focus metric discriminates sharp vs blurred luminance patterns | portable `image_quality` module | CTest verified |
| Camera and PPS/UTC clocks are explicitly aligned | `UtcMonotonicAligner` estimates UTC→host-monotonic offset and lock/jitter | CTest verified |
| Live observations are resampled to the model's fixed timebase | `FixedRateSpatialResampler` | CTest verified |
| Missing camera intervals do not bridge Transformer sequences | native runtime clears temporal queues across unmatched gaps | Implemented |
| Native inference uses an explicit ordered feature contract | Python export validation + C++ `model_contract.hpp` + TensorRT shape check | Host-verified |
| Missing GNSS/environment cannot masquerade as physical zeros in model input | invalid context serializes as `null`; inference fails closed unless both context classes are valid | Python/CTest verified |
| TensorRT deployment is bound to an exact trained artifact | checkpoint/ONNX/engine SHA-256 lineage manifest + verified launcher | Pytest verified; engine build requires Jetson/TensorRT |
| Training artifacts record dataset SHA-256, seed, runtime versions and optional deterministic-mode state | checkpoint/classical provenance manifest | Pytest verified |
| Native runtime does not emit heuristic predictions as trained inference | no engine ⇒ inference disabled; diagnostic engine is self-test only | Host-verified |
| Telemetry contract is versioned and schema-validated | JSON Schema 2020-12, `schema_version: 1`, strict date-time checking | Pytest verified |
| MQTT topic identity must match payload device identity | `topic_matches_device` | Pytest verified |
| Broker latency cannot directly block acquisition | SQLite durable producer/consumer outbox | Pytest verified at outbox boundary |
| Offline spool is bounded and records drops | SQLite max-record policy + persistent `spool_dropped` | Pytest verified |
| Cloud ingest mapping/prediction fan-out works without a live DB | injected fake-connection processor tests | Pytest verified |
| Ingest SQL named placeholders stay synchronized with the flattened telemetry row | SQL/row contract test | Pytest verified |
| Spatial test split prevents repeated-route location leakage | 500 m geographic blocks + default 30 m geodesic purge margin; sequences are re-segmented after purge | Pytest verified |
| Spatial split records its achieved physical separation | minimum train↔held-out and validation↔test haversine distances stored in checkpoint/manifest | Pytest verified |
| Classical forecast targets cannot leak across purged geography | split/purge occurs before target shifting inside final sequence groups | Pytest verified |
| Unlabeled anomalies are not silently treated as normal | no catalog ⇒ `NaN`; supervised BCE/AUROC gated on labels | Pytest verified |
| Camera temporal-offset and frame-dropout robustness can be measured | untouched-test sensitivity evaluator | Implemented + helper tests |
| Untouched-test performance exposes between-location/run variability instead of only one global average | per-group MAE + median/IQR report | Implemented + helper tests |
| Dataset adapter reports frame-alignment QA | sidecar JSON with acceptance rate and p50/p95/max timestamp error | Implemented + unit tested |
| C++ native packet/DSP paths outperform Python reference | 7-repeat benchmark in `benchmarks/results/latest.json` | **Measured on x86-64 host** |
| Typed protocol decoders reject incompatible contract versions and recover valid frames through serial noise | deterministic C++ version/noise stress test | CTest verified |
| Cloud trend baseline can be imported/tested without starting a DB loop | explicit service entry point + fake-connection timebase test | Pytest verified |
| Multimodal accuracy improves over persistence/unimodal dependencies | untouched spatial-test evaluator | Requires prepared Rail-VIVID data + trained checkpoint |
| TensorRT latency/throughput at FP32/FP16/INT8 | build/benchmark scripts | Requires NVIDIA Jetson |
| Three physical IIS3DWB devices sustain FIFO/DMA/PPS acquisition | target HAL path + electrical design | Requires assembled hardware/logic analyzer |
| Custom acquisition PCB operates electrically/thermally as designed | BOM/netlist/power design | Requires ECAD/manufacture/bench validation |

## Local validation

```bash
make validate
```

## Repeatable host performance benchmark

```bash
make benchmark
python scripts/sync_evidence.py --write
```

CI runs `python scripts/sync_evidence.py --check`, so checked-in performance claims fail validation if they drift from the benchmark JSON or the protocol/model contract.
