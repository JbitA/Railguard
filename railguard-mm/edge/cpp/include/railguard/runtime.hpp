#pragma once
#include "railguard/dsp.hpp"
#include "railguard/inference.hpp"
#include <array>
#include <cstdint>
#include <memory>
#include <span>
namespace railguard {
struct SensorVector { std::array<float,9> values{}; };
struct RuntimeResult { VibrationFeatures vibration; Prediction prediction; };
class EdgeRuntime {
public:
    explicit EdgeRuntime(std::unique_ptr<InferenceEngine> engine);
    RuntimeResult process(std::span<const float> acceleration, std::span<const float> frames_chw,
                          std::span<const float> sensors_tf, std::size_t time_steps,
                          std::size_t height, std::size_t width);
    [[nodiscard]] const InferenceEngine& engine() const noexcept { return *engine_; }
private:
    std::unique_ptr<InferenceEngine> engine_;
};
}
