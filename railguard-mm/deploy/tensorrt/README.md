# ONNX -> TensorRT deployment

The training checkpoint is exported through `ml/export_onnx.py`. The deployment wrapper contains the sensor mean/std, so the C++ runtime supplies physical-unit features rather than duplicating training normalization.

```bash
python ml/export_onnx.py models/fusion_transformer.pt
./deploy/tensorrt/build_engine.sh models/fusion_transformer.onnx models/fusion_transformer_fp16.engine
./deploy/tensorrt/benchmark_engine.sh models/fusion_transformer_fp16.engine
```

The engine has named tensors `frames`, `sensors`, `vibration`, `vision`, and `anomaly_probability`. `frames` are RGB FP32 in `[0,1]` with shape `[1,T,3,96,96]`; sensors are raw physical features `[1,T,9]`. The build profile accepts T=4..128 and optimizes T=32.

Build the native runtime on Jetson:

```bash
cmake -S edge/cpp -B build/edge-jetson -DCMAKE_BUILD_TYPE=Release \
  -DRAILGUARD_ENABLE_CUDA=ON -DRAILGUARD_ENABLE_TENSORRT=ON
cmake --build build/edge-jetson -j$(nproc)
./build/edge-jetson/railguard_edge models/fusion_transformer_fp16.engine
```

For final performance measurements, keep two benchmark modes:

1. `trtexec --noDataTransfers` measures the inference engine itself.
2. The C++ application benchmark measures end-to-end capture/preprocess/inference/telemetry latency, including transfers that have not been eliminated by the NVMM/CUDA path.

Do not copy example FPS/latency figures into the README. Commit the output produced on the actual Jetson and record JetPack/TensorRT versions, power mode, clocks, sequence length and precision.
