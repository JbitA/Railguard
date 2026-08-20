#pragma once
#include <array>
#include <cstddef>
#include <string_view>

namespace railguard {
inline constexpr std::string_view kImageMode = "monochrome_replicated_rgb";
inline constexpr std::size_t kSensorFeatureDim = 9;
inline constexpr std::array<std::string_view, kSensorFeatureDim> kSensorFeatureNames{
    "vibration_rms", "vibration_peak", "vibration_kurtosis", "crest_factor",
    "vision_motion", "vision_contrast", "speed_mps", "temperature_c", "humidity"
};
} // namespace railguard
