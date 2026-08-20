#include "railguard/inference.hpp"
#include "railguard/model_contract.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace railguard {
bool validate_prediction(const Prediction& p) noexcept {
    for (float v : p.vibration) {
        if (!std::isfinite(v) || v < 0.0f) return false;
    }
    for (float v : p.vision) {
        if (!std::isfinite(v) || v < 0.0f || v > 1.0f) return false;
    }
    return std::isfinite(p.anomaly_probability) &&
           p.anomaly_probability >= 0.0f && p.anomaly_probability <= 1.0f;
}
class ReferenceEngine final : public InferenceEngine {
public:
    Prediction infer(std::span<const float> frames, std::span<const float> sensors,
                     std::size_t, std::size_t, std::size_t) override {
        Prediction p{};
        const float s = sensors.size() < kSensorFeatureDim ? 0.0f : sensors[sensors.size() - kSensorFeatureDim];
        p.vibration = {s, s * 1.01f, s * 1.02f};
        const float vm = sensors.size() >= kSensorFeatureDim ? sensors[sensors.size() - kSensorFeatureDim + 4] : 0.0f;
        p.vision = {vm, vm, vm};
        const float mean = frames.empty() ? 0.0f
            : std::accumulate(frames.begin(), frames.end(), 0.0f) / static_cast<float>(frames.size());
        p.anomaly_probability = std::clamp(0.05f + 0.2f * std::abs(s) + 0.05f * mean, 0.0f, 1.0f);
        return p;
    }
    std::string backend() const override { return "diagnostic-reference"; }
};

std::unique_ptr<InferenceEngine> make_reference_engine() {
    return std::make_unique<ReferenceEngine>();
}

#ifndef RAILGUARD_ENABLE_TENSORRT
std::unique_ptr<InferenceEngine> make_inference_engine(const std::string& engine_path) {
    if (!engine_path.empty()) {
        throw std::runtime_error(
            "an engine path was supplied, but this binary was built without TensorRT; "
            "reconfigure with -DRAILGUARD_ENABLE_TENSORRT=ON");
    }
    throw std::runtime_error("trained inference requires an explicit deployment backend/engine");
}
#endif
}
