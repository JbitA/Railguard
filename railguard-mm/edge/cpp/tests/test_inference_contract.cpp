#include "railguard/inference.hpp"
#include <cassert>
#include <limits>

int main() {
    railguard::Prediction p{};
    p.vibration = {0.0f, 1.0f, 2.0f};
    p.vision = {0.0f, 0.5f, 1.0f};
    p.anomaly_probability = 0.7f;
    assert(railguard::validate_prediction(p));
    p.vibration[0] = -0.01f; assert(!railguard::validate_prediction(p));
    p.vibration[0] = 0.0f; p.vision[1] = 1.01f; assert(!railguard::validate_prediction(p));
    p.vision[1] = 0.5f; p.anomaly_probability = std::numeric_limits<float>::quiet_NaN(); assert(!railguard::validate_prediction(p));
    p.anomaly_probability = -0.1f; assert(!railguard::validate_prediction(p));
}
