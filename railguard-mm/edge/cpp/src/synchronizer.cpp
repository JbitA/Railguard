#include "railguard/synchronizer.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace railguard {
FrameSynchronizer::FrameSynchronizer(std::size_t capacity, double max_delta_ms)
    : capacity_(capacity), max_delta_ns_(static_cast<std::uint64_t>(std::max(0.0, max_delta_ms) * 1e6)) {
    if (capacity_ == 0) throw std::invalid_argument("frame synchronizer capacity must be non-zero");
}

void FrameSynchronizer::push(CameraFrame frame) {
    auto p = std::make_shared<CameraFrame>(std::move(frame));
    std::scoped_lock lk(mutex_);
    frames_.push_back(std::move(p));
    while (frames_.size() > capacity_) frames_.pop_front();
}

std::optional<FrameMatch> FrameSynchronizer::nearest(std::uint64_t sensor_monotonic_ns) const {
    std::scoped_lock lk(mutex_);
    if (frames_.empty() || sensor_monotonic_ns == 0) return std::nullopt;
    auto best = frames_.front();
    auto absdiff = [sensor_monotonic_ns](std::uint64_t x) {
        return x >= sensor_monotonic_ns ? x - sensor_monotonic_ns : sensor_monotonic_ns - x;
    };
    std::uint64_t best_abs = absdiff(best->monotonic_ns);
    for (const auto& f : frames_) {
        const auto d = absdiff(f->monotonic_ns);
        if (d < best_abs) { best = f; best_abs = d; }
    }
    if (best_abs > max_delta_ns_) return std::nullopt;
    const auto signed_ns = static_cast<std::int64_t>(best->monotonic_ns) - static_cast<std::int64_t>(sensor_monotonic_ns);
    return FrameMatch{best, static_cast<double>(signed_ns) / 1e6};
}

std::size_t FrameSynchronizer::size() const {
    std::scoped_lock lk(mutex_);
    return frames_.size();
}
} // namespace railguard
