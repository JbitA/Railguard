# Native edge runtime (C++20 / CUDA / TensorRT)

This is the deterministic deployment path. Python remains the reference/training environment.

## CPU/CI build

```bash
cmake -S edge/cpp -B build/edge-cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/edge-cpp -j
./build/edge-cpp/railguard_edge
./build/edge-cpp/railguard_bench 200000
```

The dependency-free build validates the binary protocol, CRC, native DSP, SPSC backpressure primitive and runtime interface on standard CI runners.

## Jetson build

```bash
cmake -S edge/cpp -B build/edge-jetson -DCMAKE_BUILD_TYPE=Release \
  -DRAILGUARD_ENABLE_CUDA=ON -DRAILGUARD_ENABLE_TENSORRT=ON
cmake --build build/edge-jetson -j$(nproc)
```

`cuda/preprocess.cu` converts BGR byte frames into normalized planar RGB directly on a CUDA stream. `tensorrt_engine.cpp` uses TensorRT named I/O tensors and `enqueueV3` for the exported multimodal model.

The next hardware integration step is to bind camera capture to NVMM/GStreamer and pass device-resident buffers to the CUDA preprocessor. The current TensorRT class accepts host spans so it is testable independently; the interface is deliberately isolated so a zero-copy buffer type can replace those spans without changing protocol/DSP code.

## Cloud handoff

The executable emits schema-compatible NDJSON. Keep MQTT and retry logic outside the latency-critical process:

```bash
./build/edge-cpp/railguard_edge --serial /dev/ttyACM0 --camera /dev/video0 | \
  python -m edge.railguard_edge.native_bridge --config edge/config.example.yaml
```

This process boundary means an MQTT reconnect or disk-spool flush cannot block serial FIFO servicing or model execution.
