#pragma once
#include "railguard/protocol.hpp"
#include <array>
#include <cstdint>
#include <optional>

namespace railguard {
struct SpatialAggregate {
    std::array<SensorFeaturePayload,3> sensors{};
    float rms{};
    float peak{};
    float kurtosis{};
    float crest_factor{};
    std::array<float,4> band_energy{};
    std::uint64_t utc_ns{};
    double sensor_skew_ms{};
    float temperature_c{};
    float humidity{};
    float latitude{};
    float longitude{};
    float speed_mps{};
    std::uint8_t context_flags{};
};

class SpatialVibrationAggregator {
public:
    explicit SpatialVibrationAggregator(double max_sensor_skew_ms = 50.0);
    std::optional<SpatialAggregate> update(const SensorFeaturePayload& payload, std::uint64_t utc_ns);
    [[nodiscard]] std::uint8_t fresh_mask() const noexcept { return fresh_mask_; }
private:
    double max_sensor_skew_ms_;
    std::array<SensorFeaturePayload,3> latest_{};
    std::array<std::uint64_t,3> timestamps_{};
    std::uint8_t valid_mask_{};
    std::uint8_t fresh_mask_{};
};
} // namespace railguard
