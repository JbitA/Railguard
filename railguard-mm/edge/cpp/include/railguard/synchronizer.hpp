#pragma once
#include "railguard/v4l2_camera.hpp"
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <optional>

namespace railguard {
struct FrameMatch {
    std::shared_ptr<const CameraFrame> frame;
    double delta_ms{}; // camera monotonic timestamp - mapped sensor monotonic timestamp
};

class FrameSynchronizer {
public:
    explicit FrameSynchronizer(std::size_t capacity = 128, double max_delta_ms = 50.0);
    void push(CameraFrame frame);
    [[nodiscard]] std::optional<FrameMatch> nearest(std::uint64_t sensor_monotonic_ns) const;
    [[nodiscard]] std::size_t size() const;
private:
    std::size_t capacity_;
    std::uint64_t max_delta_ns_;
    mutable std::mutex mutex_;
    std::deque<std::shared_ptr<CameraFrame>> frames_;
};
} // namespace railguard
