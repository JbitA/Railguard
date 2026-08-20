#pragma once

#include "railguard/spatial_fusion.hpp"
#include <cstdint>
#include <optional>
#include <vector>

namespace railguard {

class FixedRateSpatialResampler {
public:
    explicit FixedRateSpatialResampler(double period_ms = 100.0, double max_gap_periods = 3.0);
    [[nodiscard]] std::vector<SpatialAggregate> push(const SpatialAggregate& sample);
    [[nodiscard]] double period_ms() const noexcept { return static_cast<double>(period_ns_) / 1e6; }

private:
    [[nodiscard]] static SpatialAggregate interpolate(
        const SpatialAggregate& a, const SpatialAggregate& b, std::uint64_t target_ns);
    [[nodiscard]] std::uint64_t ceil_to_grid(std::uint64_t t) const noexcept;

    std::uint64_t period_ns_;
    std::uint64_t max_gap_ns_;
    std::optional<SpatialAggregate> previous_;
    std::uint64_t next_target_ns_{};
};

} // namespace railguard
