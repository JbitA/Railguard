#pragma once
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace railguard {
struct CameraFrame {
    std::vector<float> rgb_chw;
    std::uint64_t monotonic_ns{};
    std::uint64_t utc_ns{};
    float contrast{};
    float motion{};
    float sharpness{};
};
class V4L2Camera {
public:
    V4L2Camera(const std::string& device, int width=640, int height=480);
    ~V4L2Camera();
    V4L2Camera(const V4L2Camera&)=delete;V4L2Camera& operator=(const V4L2Camera&)=delete;
    bool capture(CameraFrame& out, int timeout_ms=200);
private:
    struct Impl; Impl* impl_;
};
}
