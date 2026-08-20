#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>

namespace railguard {

struct ClockAlignmentStatus {
    bool locked{};
    std::size_t samples{};
    double jitter_ms{};
    double estimated_transport_bias_ms{};
};

class UtcMonotonicAligner {
public:
    explicit UtcMonotonicAligner(std::size_t capacity = 128, std::size_t min_samples = 8);

    // Observe a GNSS/PPS-derived sensor timestamp and the host monotonic time at
    // which its packet was received.  Serial/processing latency is non-negative,
    // so high quantiles of (sensor_utc - receive_monotonic) approximate the clock
    // offset while rejecting a single impossible/outlier sample.
    void observe(std::uint64_t sensor_utc_ns, std::uint64_t receive_monotonic_ns);

    [[nodiscard]] std::optional<std::uint64_t> monotonic_from_utc(std::uint64_t sensor_utc_ns) const;
    [[nodiscard]] ClockAlignmentStatus status() const;

private:
    struct Sample { std::int64_t offset_ns{}; };
    [[nodiscard]] std::optional<std::int64_t> offset_estimate_ns() const;
    [[nodiscard]] static std::int64_t quantile(std::deque<Sample> samples, double q);

    std::size_t capacity_;
    std::size_t min_samples_;
    std::deque<Sample> samples_;
};

std::uint64_t steady_now_ns();

} // namespace railguard
