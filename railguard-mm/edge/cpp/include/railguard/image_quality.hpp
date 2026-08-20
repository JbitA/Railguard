#pragma once

#include <span>

namespace railguard {

// Variance of a 4-neighbour discrete Laplacian, normalized to the same scale
// used by the telemetry camera-quality metric. Input is a row-major luma
// image in nominal [0,255] units.
float laplacian_sharpness(std::span<const float> luma, int width, int height);

} // namespace railguard
