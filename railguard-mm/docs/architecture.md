# Architecture rationale

## Functional decomposition

RailGuard-MM is split into four timing/security domains:

1. **Sensor domain** — high-rate vibration and camera acquisition.
2. **Edge domain** — C++20 synchronization/perception/inference plus a separately buffered telemetry/network boundary.
3. **Cloud data domain** — durable time-series/object storage and model execution.
4. **Presentation domain** — read-only API access for browser visualization.

This separation prevents a web or networking failure from disturbing sensor acquisition and prevents browser clients from receiving storage credentials.

## Timing model

The STM32 uses a monotonic hardware timer disciplined by GNSS PPS. Each vibration packet contains:

- PPS epoch counter;
- microseconds since last PPS;
- sensor sample counter;
- sequence number;
- CRC32.

The camera is captured through V4L2 on the Jetson and aligned to the MCU clock through the PPS epoch. The native runtime keeps acquisition and inference in C++ so Python interpreter latency is not part of the real-time data path. The edge service stores both acquisition time and publish time. Cloud forecasts store `issued_at` and `target_ts` separately.

## Edge/cloud bandwidth strategy

Raw IIS3DWB data can be hundreds of kilobytes per second per sensor. The default cloud stream therefore sends window features at 10 Hz while retaining raw windows locally. Event-triggered raw windows and frames are copied to object storage.

## Resilience

- MQTT QoS 1 for scalar telemetry.
- Sequence numbers expose loss/reordering.
- Durable SQLite MQTT outbox on the Jetson for disconnected scalar telemetry.
- NVMe-backed event cache for image artifacts that could not yet be uploaded.
- Idempotent inserts keyed by `(device_id, ts)`.
- Predictions keyed by `(device_id, issued_at, target_ts, model_version)`.

## Security boundary

Recommended deployed configuration:

- TLS 1.2+ for MQTT and HTTPS;
- per-device MQTT credentials/certificates;
- no inbound port to the edge node;
- FastAPI is the only public data interface;
- database and object store remain private to the cloud network;
- secrets are injected as environment variables, not committed.

## Native runtime concurrency

The optimized edge runtime separates I/O and compute responsibilities:

```text
STM32 serial -> C++ CRC stream decoder -> temporal sensor ring ─┐
                                                                ├-> TensorRT -> telemetry
V4L2 mmap camera -> frame conversion -> temporal image ring ────┘
```

The reusable `SpscRing` primitive provides bounded single-producer/single-consumer backpressure for the threaded hardware integration. The reference executable currently uses a camera worker plus the serial/inference loop; production profiling can split preprocessing and inference into dedicated workers when measurements justify the extra scheduling complexity.
