#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

typedef struct { int16_t x,y,z; } rg_xyz_i16_t;
typedef struct {
    float axis_rms[3];
    float rms;
    float peak;
    float kurtosis;
    float crest_factor;
    float band_energy[4];
} rg_vibration_features_t;

bool rg_compute_vibration_features(const rg_xyz_i16_t *samples, size_t count, float sample_rate_hz, rg_vibration_features_t *out);
#ifdef __cplusplus
}
#endif
