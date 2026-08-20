#include "environment_decode.h"
#include <stddef.h>

uint8_t rg_sensirion_crc8(const uint8_t *data, uint16_t length) {
    uint8_t crc = 0xffu;
    for (uint16_t i = 0; i < length; ++i) {
        crc ^= data[i];
        for (unsigned bit = 0; bit < 8u; ++bit) {
            if ((crc & 0x80u) != 0u) {
                crc = (uint8_t)(((uint16_t)crc << 1u) ^ 0x31u);
            } else {
                crc = (uint8_t)((uint16_t)crc << 1u);
            }
        }
    }
    return crc;
}

bool rg_sht4x_decode(const uint8_t response[6], float *temperature_c, float *humidity_rh) {
    if (response == NULL || temperature_c == NULL || humidity_rh == NULL) {
        return false;
    }
    if (rg_sensirion_crc8(response, 2u) != response[2] ||
        rg_sensirion_crc8(response + 3, 2u) != response[5]) {
        return false;
    }

    const uint16_t st = (uint16_t)(((uint16_t)response[0] << 8u) | response[1]);
    const uint16_t srh = (uint16_t)(((uint16_t)response[3] << 8u) | response[4]);
    *temperature_c = -45.0f + 175.0f * (float)st / 65535.0f;
    float humidity_percent = -6.0f + 125.0f * (float)srh / 65535.0f;
    if (humidity_percent < 0.0f) humidity_percent = 0.0f;
    if (humidity_percent > 100.0f) humidity_percent = 100.0f;
    *humidity_rh = humidity_percent / 100.0f;
    return true;
}
