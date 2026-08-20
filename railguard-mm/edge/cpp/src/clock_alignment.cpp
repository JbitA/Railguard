#include "railguard/clock_alignment.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace railguard {

UtcMonotonicAligner::UtcMonotonicAligner(std::size_t capacity, std::size_t min_samples)
    : capacity_(capacity), min_samples_(min_samples) {
    if (capacity_ < 4) throw std::invalid_argument("clock aligner capacity must be >= 4");
    if (min_samples_ < 2 || min_samples_ > capacity_) throw std::invalid_argument("invalid clock aligner min_samples");
}

void UtcMonotonicAligner::observe(std::uint64_t sensor_utc_ns, std::uint64_t receive_monotonic_ns) {
    if (sensor_utc_ns == 0 || receive_monotonic_ns == 0) return;
    if (sensor_utc_ns > static_cast<std::uint64_t>(INT64_MAX) || receive_monotonic_ns > static_cast<std::uint64_t>(INT64_MAX)) return;
    const auto offset = static_cast<std::int64_t>(sensor_utc_ns) - static_cast<std::int64_t>(receive_monotonic_ns);
    samples_.push_back({offset});
    while (samples_.size() > capacity_) samples_.pop_front();
}

std::int64_t UtcMonotonicAligner::quantile(std::deque<Sample> samples, double q) {
    std::vector<std::int64_t> values;
    values.reserve(samples.size());
    for (const auto& s : samples) values.push_back(s.offset_ns);
    std::sort(values.begin(), values.end());
    const auto idx = static_cast<std::size_t>(std::llround(q * static_cast<double>(values.size() - 1)));
    return values[std::min(idx, values.size() - 1)];
}

std::optional<std::int64_t> UtcMonotonicAligner::offset_estimate_ns() const {
    if (samples_.size() < min_samples_) return std::nullopt;
    // 90th percentile: close to the minimum one-way transport delay, but robust
    // against a single spuriously large offset unlike a raw max estimator.
    return quantile(samples_, 0.90);
}

std::optional<std::uint64_t> UtcMonotonicAligner::monotonic_from_utc(std::uint64_t sensor_utc_ns) const {
    const auto est = offset_estimate_ns();
    if (!est || sensor_utc_ns > static_cast<std::uint64_t>(INT64_MAX)) return std::nullopt;
    const auto mono = static_cast<std::int64_t>(sensor_utc_ns) - *est;
    if (mono <= 0) return std::nullopt;
    return static_cast<std::uint64_t>(mono);
}

ClockAlignmentStatus UtcMonotonicAligner::status() const {
    ClockAlignmentStatus out{};
    out.samples = samples_.size();
    out.locked = samples_.size() >= min_samples_;
    if (samples_.size() >= 2) {
        const auto p10 = quantile(samples_, 0.10);
        const auto p90 = quantile(samples_, 0.90);
        out.jitter_ms = static_cast<double>(p90 - p10) / 1e6;
        // Difference between the 90th percentile offset and the median approximates
        // the latency bias removed by using a high offset quantile.
        const auto p50 = quantile(samples_, 0.50);
        out.estimated_transport_bias_ms = static_cast<double>(p90 - p50) / 1e6;
    }
    return out;
}

std::uint64_t steady_now_ns() {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

} // namespace railguard
