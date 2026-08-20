#pragma once
#include <array>
#include <span>
namespace railguard {
struct VibrationFeatures {
    float rms{};
    float peak{};
    float kurtosis{};
    float crest_factor{};
    std::array<float,4> band_energy{};
};
VibrationFeatures vibration_features(std::span<const float> samples, float sample_rate_hz = 2000.0f) noexcept;
}
