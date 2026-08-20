#include "railguard/image_quality.hpp"

#include <algorithm>
#include <cstddef>
#include <stdexcept>

namespace railguard {

float laplacian_sharpness(std::span<const float> luma, const int width, const int height) {
    if (width < 3 || height < 3) {
        return 0.0f;
    }
    const auto required = static_cast<std::size_t>(width) * static_cast<std::size_t>(height);
    if (luma.size() != required) {
        throw std::invalid_argument("luma size does not match image dimensions");
    }

    double sum = 0.0;
    double sum_sq = 0.0;
    std::size_t count = 0;
    for (int y = 1; y < height - 1; ++y) {
        for (int x = 1; x < width - 1; ++x) {
            const auto k = static_cast<std::size_t>(y * width + x);
            const double lap = 4.0 * luma[k] - luma[k - 1] - luma[k + 1]
                             - luma[k - static_cast<std::size_t>(width)]
                             - luma[k + static_cast<std::size_t>(width)];
            sum += lap;
            sum_sq += lap * lap;
            ++count;
        }
    }

    if (count == 0) {
        return 0.0f;
    }
    const double mean = sum / static_cast<double>(count);
    const double variance = std::max(0.0, sum_sq / static_cast<double>(count) - mean * mean);
    return static_cast<float>(variance / 1000.0);
}

} // namespace railguard
