#include "railguard/camera_preprocess.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <linux/videodev2.h>
#include <vector>

int main() {
    // Y12: 2x2 little-endian 12-bit samples with padding at the end of each row.
    std::vector<std::uint8_t> y12(12, 0);
    const std::uint16_t values[4] = {0, 4095, 2048, 1024};
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            const auto off = static_cast<std::size_t>(y * 6 + x * 2);
            const auto v = values[y * 2 + x];
            y12[off] = static_cast<std::uint8_t>(v & 0xffu);
            y12[off + 1] = static_cast<std::uint8_t>(v >> 8u);
        }
    }
    auto mono = railguard::preprocess_v4l2_frame(y12, V4L2_PIX_FMT_Y12, 2, 2, 6, 2, 2);
    assert(mono.rgb_chw.size() == 12);
    assert(std::abs(mono.rgb_chw[0] - 0.0f) < 1e-6f);
    assert(std::abs(mono.rgb_chw[1] - 1.0f) < 1e-6f);
    // Monochrome is deliberately replicated across all three model channels.
    assert(std::abs(mono.rgb_chw[1] - mono.rgb_chw[5]) < 1e-6f);
    assert(std::abs(mono.rgb_chw[1] - mono.rgb_chw[9]) < 1e-6f);

    std::vector<float> previous = mono.luma;
    y12[0] = 0xff; y12[1] = 0x0f;
    auto changed = railguard::preprocess_v4l2_frame(y12, V4L2_PIX_FMT_Y12, 2, 2, 6, 2, 2, previous);
    assert(changed.motion > 0.0f);

    // GREY respects negotiated stride rather than assuming tightly packed rows.
    std::vector<std::uint8_t> grey{0, 255, 99, 99, 128, 64, 99, 99};
    auto g = railguard::preprocess_v4l2_frame(grey, V4L2_PIX_FMT_GREY, 2, 2, 4, 2, 2);
    assert(std::abs(g.rgb_chw[1] - 1.0f) < 1e-6f);
    assert(std::abs(g.rgb_chw[2] - (128.0f / 255.0f)) < 1e-6f);

    // YUYV remains supported for development cameras / the color variant.
    std::vector<std::uint8_t> yuyv{16, 128, 235, 128};
    auto c = railguard::preprocess_v4l2_frame(yuyv, V4L2_PIX_FMT_YUYV, 2, 1, 4, 2, 1);
    assert(c.rgb_chw[0] < c.rgb_chw[1]);
    return 0;
}
