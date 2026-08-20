#pragma once
#include <stddef.h>
#include <stdint.h>
#include "dsp_features.h"
#ifdef __cplusplus
extern "C" {
#endif
#define RG_PACKET_VERSION 2u
#define RG_PACKET_TYPE_FEATURES 1u
#define RG_PACKET_TYPE_SENSOR_FEATURES 2u
#define RG_FLAG_GNSS_VALID 0x01u
#define RG_FLAG_ENV_VALID 0x02u
typedef struct { float ax_rms, ay_rms, az_rms, temperature_c, latitude, longitude, speed_mps; } rg_feature_payload_t;
typedef struct {
    uint8_t sensor_id;
    uint8_t flags;
    uint16_t window_samples;
    float sample_rate_hz;
    float axis_rms[3];
    float rms, peak, kurtosis, crest_factor;
    float band_energy[4];
    float temperature_c, humidity, latitude, longitude, speed_mps;
} rg_sensor_feature_payload_t;
typedef struct { uint32_t sequence, pps_epoch, sub_us; } rg_timestamp_t;
uint32_t rg_crc32_ieee(const uint8_t *data, size_t len);
size_t rg_encode_feature_packet(uint8_t *dst, size_t capacity, const rg_timestamp_t *ts, const rg_feature_payload_t *payload);
size_t rg_encode_sensor_feature_packet(uint8_t *dst, size_t capacity, const rg_timestamp_t *ts, const rg_sensor_feature_payload_t *payload);
#ifdef __cplusplus
}
#endif
