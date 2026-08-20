#pragma once
#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
namespace railguard {
struct Prediction {
    std::array<float,3> vibration{};
    std::array<float,3> vision{};
    float anomaly_probability{};
};
[[nodiscard]] bool validate_prediction(const Prediction&) noexcept;
class InferenceEngine {
public:
    virtual ~InferenceEngine() = default;
    virtual Prediction infer(std::span<const float> frames_chw, std::span<const float> sensors_tf,
                             std::size_t time_steps, std::size_t height, std::size_t width) = 0;
    [[nodiscard]] virtual std::string backend() const = 0;
};
std::unique_ptr<InferenceEngine> make_inference_engine(const std::string& engine_path);
std::unique_ptr<InferenceEngine> make_reference_engine();
}
