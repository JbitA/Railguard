#include "railguard/v4l2_camera.hpp"
#include "railguard/camera_preprocess.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cerrno>
#include <cstdint>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <poll.h>
#include <stdexcept>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <vector>

namespace railguard {
namespace {

int xioctl(const int fd, const unsigned long request, void* arg) {
    int result;
    do {
        result = ioctl(fd, request, arg);
    } while (result < 0 && errno == EINTR);
    return result;
}

} // namespace

struct V4L2Camera::Impl {
    struct Buffer {
        void* ptr{};
        std::size_t bytes{};
    };

    int fd{-1};
    int width{};
    int height{};
    std::uint32_t pixel_format{};
    std::size_t bytes_per_line{};
    std::vector<Buffer> buffers;
    std::vector<float> previous_luma;
};

V4L2Camera::V4L2Camera(const std::string& device, const int width, const int height)
    : impl_(new Impl) {
    auto& impl = *impl_;
    impl.fd = open(device.c_str(), O_RDWR | O_NONBLOCK);
    if (impl.fd < 0) {
        throw std::runtime_error("open V4L2 camera failed");
    }

    v4l2_format format{};
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = width;
    format.fmt.pix.height = height;
    // The selected See3CAM_50CUGM is monochrome and documents Y12 output.
    // V4L2 may negotiate another supported format, so inspect the returned FOURCC
    // and reject anything the deterministic preprocessing path cannot decode.
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_Y12;
    format.fmt.pix.field = V4L2_FIELD_ANY;
    if (xioctl(impl.fd, VIDIOC_S_FMT, &format) < 0) {
        throw std::runtime_error("VIDIOC_S_FMT failed");
    }
    impl.width = static_cast<int>(format.fmt.pix.width);
    impl.height = static_cast<int>(format.fmt.pix.height);
    impl.pixel_format = format.fmt.pix.pixelformat;
    impl.bytes_per_line = format.fmt.pix.bytesperline;
    if (impl.pixel_format != V4L2_PIX_FMT_Y12 &&
        impl.pixel_format != V4L2_PIX_FMT_GREY &&
        impl.pixel_format != V4L2_PIX_FMT_YUYV) {
        throw std::runtime_error("camera negotiated unsupported pixel format");
    }
    const std::size_t minimum_stride = impl.pixel_format == V4L2_PIX_FMT_GREY
        ? static_cast<std::size_t>(impl.width)
        : static_cast<std::size_t>(impl.width) * 2u;
    if (impl.bytes_per_line == 0) impl.bytes_per_line = minimum_stride;
    if (impl.bytes_per_line < minimum_stride) {
        throw std::runtime_error("camera negotiated invalid bytes-per-line");
    }

    v4l2_requestbuffers request{};
    request.count = 4;
    request.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    request.memory = V4L2_MEMORY_MMAP;
    if (xioctl(impl.fd, VIDIOC_REQBUFS, &request) < 0 || request.count < 2) {
        throw std::runtime_error("VIDIOC_REQBUFS failed");
    }

    impl.buffers.resize(request.count);
    for (unsigned index = 0; index < request.count; ++index) {
        v4l2_buffer buffer{};
        buffer.type = request.type;
        buffer.memory = request.memory;
        buffer.index = index;
        if (xioctl(impl.fd, VIDIOC_QUERYBUF, &buffer) < 0) {
            throw std::runtime_error("VIDIOC_QUERYBUF failed");
        }

        auto* mapped = mmap(nullptr, buffer.length, PROT_READ | PROT_WRITE, MAP_SHARED, impl.fd, buffer.m.offset);
        if (mapped == MAP_FAILED) {
            throw std::runtime_error("mmap camera buffer failed");
        }
        impl.buffers[index] = {mapped, buffer.length};

        if (xioctl(impl.fd, VIDIOC_QBUF, &buffer) < 0) {
            throw std::runtime_error("VIDIOC_QBUF failed");
        }
    }

    v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(impl.fd, VIDIOC_STREAMON, &type) < 0) {
        throw std::runtime_error("VIDIOC_STREAMON failed");
    }
}

V4L2Camera::~V4L2Camera() {
    if (impl_ == nullptr) {
        return;
    }
    auto& impl = *impl_;
    if (impl.fd >= 0) {
        v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        xioctl(impl.fd, VIDIOC_STREAMOFF, &type);
        for (auto& buffer : impl.buffers) {
            if (buffer.ptr != nullptr && buffer.ptr != MAP_FAILED) {
                munmap(buffer.ptr, buffer.bytes);
            }
        }
        close(impl.fd);
    }
    delete impl_;
}

bool V4L2Camera::capture(CameraFrame& out, const int timeout_ms) {
    auto& impl = *impl_;
    pollfd descriptor{impl.fd, POLLIN, 0};
    if (poll(&descriptor, 1, timeout_ms) <= 0) {
        return false;
    }

    v4l2_buffer buffer{};
    buffer.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buffer.memory = V4L2_MEMORY_MMAP;
    if (xioctl(impl.fd, VIDIOC_DQBUF, &buffer) < 0) {
        return false;
    }
    if (buffer.index >= impl.buffers.size()) {
        throw std::runtime_error("V4L2 returned out-of-range buffer index");
    }
    bool requeued = false;
    const auto requeue = [&]() {
        if (!requeued) {
            if (xioctl(impl.fd, VIDIOC_QBUF, &buffer) < 0) {
                throw std::runtime_error("VIDIOC_QBUF failed after capture");
            }
            requeued = true;
        }
    };

    constexpr int output_width = 96;
    constexpr int output_height = 96;

    const auto* source = static_cast<const std::uint8_t*>(impl.buffers[buffer.index].ptr);
    const std::size_t available = std::min<std::size_t>(
        buffer.bytesused ? static_cast<std::size_t>(buffer.bytesused) : impl.buffers[buffer.index].bytes,
        impl.buffers[buffer.index].bytes);
    try {
        auto processed = preprocess_v4l2_frame(
            std::span<const std::uint8_t>(source, available), impl.pixel_format,
            impl.width, impl.height, impl.bytes_per_line, output_width, output_height,
            impl.previous_luma);
        out.rgb_chw = std::move(processed.rgb_chw);
        out.contrast = processed.contrast;
        out.motion = processed.motion;
        out.sharpness = processed.sharpness;
        impl.previous_luma = std::move(processed.luma);
    } catch (...) {
        requeue();
        throw;
    }

    const auto steady_now = std::chrono::steady_clock::now();
    const auto system_now = std::chrono::system_clock::now();
    const auto dequeue_mono_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(steady_now.time_since_epoch()).count());
    const auto system_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(system_now.time_since_epoch()).count());

    // Preserve the kernel frame timestamp in the host monotonic clock domain.
    // Sensor UTC is independently aligned into this domain by UtcMonotonicAligner.
    if ((buffer.flags & V4L2_BUF_FLAG_TIMESTAMP_MONOTONIC) != 0) {
        out.monotonic_ns = static_cast<std::uint64_t>(buffer.timestamp.tv_sec) * 1'000'000'000ull
                         + static_cast<std::uint64_t>(buffer.timestamp.tv_usec) * 1000ull;
    } else {
        // Driver fallback: dequeue time is usable but has worse timing uncertainty.
        out.monotonic_ns = dequeue_mono_ns;
    }

    // Approximate UTC is diagnostics-only. Multimodal fusion uses monotonic_ns.
    const auto wall_to_mono_offset = static_cast<std::int64_t>(system_ns)
                                   - static_cast<std::int64_t>(dequeue_mono_ns);
    out.utc_ns = static_cast<std::uint64_t>(
        static_cast<std::int64_t>(out.monotonic_ns) + wall_to_mono_offset);

    requeue();
    return true;
}

} // namespace railguard
