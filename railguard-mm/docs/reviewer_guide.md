# RailGuard-MM: 10-minute reviewer guide

This page is the shortest path through the repository for an autonomy, embedded, edge-AI or multimodal-ML reviewer. It separates implemented evidence from hardware/model measurements that still require the target platform or public-data training run.

## 1. System boundary

Start with the architecture in [`README.md`](../README.md). The design intentionally assigns deterministic sensor timing to the STM32, perception/inference to native C++ on Jetson, durable transport to the edge/cloud bridge, temporal persistence to TimescaleDB and experiment/model development to Python/PyTorch.

## 2. Embedded acquisition

Read these files in order:

- [`firmware/stm32_hal/Core/Src/railguard_acquisition.c`](../firmware/stm32_hal/Core/Src/railguard_acquisition.c) — IIS3DWB FIFO watermark and burst-DMA acquisition boundary.
- [`firmware/stm32_hal/Core/Src/dsp_features.c`](../firmware/stm32_hal/Core/Src/dsp_features.c) — 512-sample statistical/spectral vibration features.
- [`firmware/stm32_hal/Core/Src/gnss_stream.c`](../firmware/stm32_hal/Core/Src/gnss_stream.c) and [`gnss_time.c`](../firmware/stm32_hal/Core/Src/gnss_time.c) — chunk-safe NMEA framing and UTC/GNSS context.
- [`firmware/stm32_hal/Core/Src/transport_queue.c`](../firmware/stm32_hal/Core/Src/transport_queue.c) — bounded non-blocking CDC transport queue.
- [`firmware/stm32_hal/host_test.c`](../firmware/stm32_hal/host_test.c) — portable host verification built with warnings-as-errors.

Hardware intent and remaining physical-validation boundaries are in [`hardware/electrical_design.md`](../hardware/electrical_design.md), [`hardware/power_budget.md`](../hardware/power_budget.md) and the [verification matrix](verification.md). The [design FMEA](../hardware/fmea.md) and [bring-up plan](../hardware/bringup_and_verification.md) show how those assumptions would be challenged on physical hardware.

## 3. Native edge runtime

The important C++20 paths are:

- [`edge/cpp/src/protocol.cpp`](../edge/cpp/src/protocol.cpp) — CRC framing, semantic payload validation and noise-resilient stream recovery.
- [`edge/cpp/src/clock_alignment.cpp`](../edge/cpp/src/clock_alignment.cpp) — GNSS/UTC to Linux-monotonic clock-domain alignment.
- [`edge/cpp/src/spatial_fusion.cpp`](../edge/cpp/src/spatial_fusion.cpp) — distinct 3-sensor observations and skew rejection.
- [`edge/cpp/src/resampler.cpp`](../edge/cpp/src/resampler.cpp) — conversion from ~19 ms feature windows to the model's fixed 100 ms timebase.
- [`edge/cpp/src/synchronizer.cpp`](../edge/cpp/src/synchronizer.cpp) — timestamp-nearest camera association.
- [`edge/cpp/src/v4l2_camera.cpp`](../edge/cpp/src/v4l2_camera.cpp), [`camera_preprocess.cpp`](../edge/cpp/src/camera_preprocess.cpp) and [`image_quality.cpp`](../edge/cpp/src/image_quality.cpp) — V4L2 mmap acquisition plus host-tested Y12/GREY/YUYV preprocessing, motion/contrast and sharpness/focus QA.
- [`edge/cpp/src/main.cpp`](../edge/cpp/src/main.cpp) — end-to-end runtime orchestration.
- [`edge/cpp/src/tensorrt_engine.cpp`](../edge/cpp/src/tensorrt_engine.cpp) — optional deployment backend when built on the NVIDIA target.

The current live V4L2 path performs resize/color conversion on CPU before TensorRT. NVMM/DMA-BUF→CUDA preprocessing is intentionally documented as future target optimization, not as an already measured zero-copy path.

## 4. ML experiment integrity

The highest-value files are:

- [`ml/railguard_ml/contracts.py`](../ml/railguard_ml/contracts.py) — exact ordered nine-feature deployment contract.
- [`ml/railguard_ml/splits.py`](../ml/railguard_ml/splits.py) — default 500 m geographic partition plus 30 m geodesic purge margin, physical-separation diagnostics and post-purge sequence re-segmentation.
- [`ml/train_multimodal.py`](../ml/train_multimodal.py) — training, checkpoint selection, anomaly-label gating and provenance capture.
- [`ml/evaluate_multimodal.py`](../ml/evaluate_multimodal.py) — untouched-test persistence/fusion dependency comparisons, temporal-offset/dropout sensitivity and per-test-group variability.
- [`ml/prepare_rail_vivid.py`](../ml/prepare_rail_vivid.py) — Rail-VIVID alignment/unit/timezone QA and optional anomaly-catalog association.
- [`ml/railguard_ml/provenance.py`](../ml/railguard_ml/provenance.py) — processed-table SHA-256 and runtime/training provenance.
- [`scripts/verify_model_manifest.py`](../scripts/verify_model_manifest.py) — deployment-time engine SHA-256 verification and bound model identity.

The repository does **not** claim multimodal accuracy improvement until a trained checkpoint is evaluated on the untouched spatial test blocks. The checkpoint also records the row-level geographic purge margin and achieved minimum physical separations. That is a deliberate evidence boundary.

## 5. Cloud and dashboard

- [`edge/railguard_edge/native_bridge.py`](../edge/railguard_edge/native_bridge.py) — durable native-process boundary; malformed NDJSON is quarantined rather than killing acquisition.
- [`cloud/ingestor/processor.py`](../cloud/ingestor/processor.py) — schema-validated transform/upsert and prediction fan-out.
- [`schemas/telemetry.schema.json`](../schemas/telemetry.schema.json) — versioned telemetry contract.
- [`cloud/api/app/main.py`](../cloud/api/app/main.py) — time-series/residual API.
- [`web/src/App.tsx`](../web/src/App.tsx) — measured/predicted values, synchronization quality, three-sensor spatial RMS, sharpness and edge-health dashboard.

## 6. Evidence to inspect

Run:

```bash
./scripts/validate_all.sh
make benchmark
python scripts/sync_evidence.py --check
```

The host benchmark source of truth is [`benchmarks/results/latest.json`](../benchmarks/results/latest.json). The generated tables in the README and [`docs/performance.md`](performance.md) are checked against it in CI.

## 7. What still requires external evidence

Three claims are intentionally left as hardware/data-required rather than inferred from local CI:

1. **Rail-VIVID model quality** — train on prepared public data and publish untouched spatial-test metrics, group variability and robustness ablations.
2. **Jetson TensorRT performance** — measure FP32/FP16/INT8 latency, throughput, memory, power and accuracy change on the actual Jetson.
3. **Physical acquisition-board validation** — verify three IIS3DWB devices, PPS/DMA timing, USB sustained throughput, FIFO overrun rate, power/thermal behavior and EMC-oriented design assumptions on hardware.

A reviewer should treat those as the next empirical milestones, not completed measurements. For the deliberately skeptical finding/resolution table, continue with [`adversarial_review.md`](adversarial_review.md).
