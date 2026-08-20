#include "railguard/camera_preprocess.hpp"
#include "railguard/image_quality.hpp"

#include <algorithm>
#include <cmath>
#include <linux/videodev2.h>
#include <stdexcept>

namespace railguard {
namespace {
float clamp255(float value) { return std::clamp(value, 0.0f, 255.0f); }

float sample_luma(std::span<const std::uint8_t> src, std::uint32_t fmt,
                  int x, int y, std::size_t stride) {
    const auto row = static_cast<std::size_t>(y) * stride;
    if (fmt == V4L2_PIX_FMT_Y12) {
        const auto off = row + static_cast<std::size_t>(x) * 2u;
        if (off + 1u >= src.size()) throw std::runtime_error("truncated Y12 camera buffer");
        const std::uint16_t raw = static_cast<std::uint16_t>(src[off]) |
                                  (static_cast<std::uint16_t>(src[off + 1u]) << 8u);
        return static_cast<float>(raw & 0x0FFFu) * (255.0f / 4095.0f);
    }
    if (fmt == V4L2_PIX_FMT_GREY) {
        const auto off = row + static_cast<std::size_t>(x);
        if (off >= src.size()) throw std::runtime_error("truncated GREY camera buffer");
        return static_cast<float>(src[off]);
    }
    if (fmt == V4L2_PIX_FMT_YUYV) {
        const int even_x = x & ~1;
        const auto off = row + static_cast<std::size_t>(even_x) * 2u;
        if (off + 3u >= src.size()) throw std::runtime_error("truncated YUYV camera buffer");
        return static_cast<float>((x & 1) ? src[off + 2u] : src[off]);
    }
    throw std::runtime_error("unsupported V4L2 pixel format");
}

void sample_rgb(std::span<const std::uint8_t> src, std::uint32_t fmt,
                int x, int y, std::size_t stride, float& r, float& g, float& b) {
    if (fmt == V4L2_PIX_FMT_Y12 || fmt == V4L2_PIX_FMT_GREY) {
        const float l = sample_luma(src, fmt, x, y, stride);
        r = g = b = l;
        return;
    }
    if (fmt == V4L2_PIX_FMT_YUYV) {
        const int even_x = x & ~1;
        const auto off = static_cast<std::size_t>(y) * stride + static_cast<std::size_t>(even_x) * 2u;
        if (off + 3u >= src.size()) throw std::runtime_error("truncated YUYV camera buffer");
        const float yy = static_cast<float>((x & 1) ? src[off + 2u] : src[off]);
        const float u = static_cast<float>(src[off + 1u]) - 128.0f;
        const float v = static_cast<float>(src[off + 3u]) - 128.0f;
        r = clamp255(yy + 1.402f * v);
        g = clamp255(yy - 0.344136f * u - 0.714136f * v);
        b = clamp255(yy + 1.772f * u);
        return;
    }
    throw std::runtime_error("unsupported V4L2 pixel format");
}

std::size_t minimum_stride(std::uint32_t fmt, int width) {
    if (fmt == V4L2_PIX_FMT_Y12 || fmt == V4L2_PIX_FMT_YUYV) return static_cast<std::size_t>(width) * 2u;
    if (fmt == V4L2_PIX_FMT_GREY) return static_cast<std::size_t>(width);
    throw std::runtime_error("unsupported V4L2 pixel format");
}
}

CameraPreprocessResult preprocess_v4l2_frame(
    std::span<const std::uint8_t> source, std::uint32_t pixel_format,
    int source_width, int source_height, std::size_t bytes_per_line,
    int output_width, int output_height, std::span<const float> previous_luma) {
    if (source_width <= 0 || source_height <= 0 || output_width <= 0 || output_height <= 0)
        throw std::invalid_argument("camera dimensions must be positive");
    const auto min_stride = minimum_stride(pixel_format, source_width);
    if (bytes_per_line < min_stride) throw std::invalid_argument("camera stride is smaller than pixel format requires");
    if (source.size() < bytes_per_line * static_cast<std::size_t>(source_height))
        throw std::invalid_argument("camera buffer is smaller than negotiated frame layout");

    const auto pixels = static_cast<std::size_t>(output_width) * static_cast<std::size_t>(output_height);
    CameraPreprocessResult out;
    out.rgb_chw.assign(3u * pixels, 0.0f);
    out.luma.assign(pixels, 0.0f);

    double mean = 0.0;
    double second_moment = 0.0;
    double motion = 0.0;
    const bool have_previous = previous_luma.size() == pixels;

    for (int y = 0; y < output_height; ++y) {
        const int sy = y * source_height / output_height;
        for (int x = 0; x < output_width; ++x) {
            const int sx = x * source_width / output_width;
            float r{}, g{}, b{};
            sample_rgb(source, pixel_format, sx, sy, bytes_per_line, r, g, b);
            const float luma = sample_luma(source, pixel_format, sx, sy, bytes_per_line);
            const auto i = static_cast<std::size_t>(y * output_width + x);
            out.rgb_chw[i] = r / 255.0f;
            out.rgb_chw[pixels + i] = g / 255.0f;
            out.rgb_chw[2u * pixels + i] = b / 255.0f;
            out.luma[i] = luma;
            mean += luma;
            second_moment += static_cast<double>(luma) * luma;
            if (have_previous) motion += std::abs(luma - previous_luma[i]);
        }
    }
    const double n = static_cast<double>(pixels);
    mean /= n;
    out.contrast = static_cast<float>(std::sqrt(std::max(0.0, second_moment / n - mean * mean)) / 64.0);
    out.motion = have_previous ? static_cast<float>(motion / n / 255.0) : 0.0f;
    out.sharpness = laplacian_sharpness(out.luma, output_width, output_height);
    return out;
}

} // namespace railguard
