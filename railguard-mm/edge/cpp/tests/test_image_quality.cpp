#include "railguard/image_quality.hpp"

#include <cassert>
#include <cmath>
#include <vector>

namespace {
std::vector<float> box_blur(const std::vector<float>& input, int width, int height, int radius) {
    std::vector<float> out(input.size(), 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            double sum = 0.0;
            int n = 0;
            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    const int xx = x + dx;
                    const int yy = y + dy;
                    if (xx >= 0 && xx < width && yy >= 0 && yy < height) {
                        sum += input[static_cast<std::size_t>(yy * width + xx)];
                        ++n;
                    }
                }
            }
            out[static_cast<std::size_t>(y * width + x)] = static_cast<float>(sum / n);
        }
    }
    return out;
}
}

int main() {
    constexpr int width = 32;
    constexpr int height = 32;
    std::vector<float> sharp(static_cast<std::size_t>(width * height), 0.0f);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            sharp[static_cast<std::size_t>(y * width + x)] = ((x / 4 + y / 4) % 2) ? 255.0f : 0.0f;
        }
    }
    const auto blurred = box_blur(sharp, width, height, 2);
    const float sharp_score = railguard::laplacian_sharpness(sharp, width, height);
    const float blurred_score = railguard::laplacian_sharpness(blurred, width, height);
    assert(std::isfinite(sharp_score));
    assert(std::isfinite(blurred_score));
    assert(sharp_score > blurred_score * 4.0f);
    assert(railguard::laplacian_sharpness({}, 0, 0) == 0.0f);
}
