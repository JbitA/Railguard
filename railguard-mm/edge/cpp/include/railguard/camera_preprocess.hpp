#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace railguard {

struct CameraPreprocessResult {
    std::vector<float> rgb_chw;
    std::vector<float> luma;
    float contrast{};
    float motion{};
    float sharpness{};
};

// Convert a negotiated V4L2 frame into the model's normalized RGB-CHW tensor.
// Monochrome inputs are replicated into RGB so the live camera contract matches
// the RGB tensors used by the training pipeline without inventing chroma.
CameraPreprocessResult preprocess_v4l2_frame(
    std::span<const std::uint8_t> source,
    std::uint32_t pixel_format,
    int source_width,
    int source_height,
    std::size_t bytes_per_line,
    int output_width,
    int output_height,
    std::span<const float> previous_luma = {});

} // namespace railguard
