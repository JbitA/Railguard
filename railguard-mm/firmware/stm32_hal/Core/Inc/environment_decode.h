#pragma once
#include <stdbool.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
uint8_t rg_sensirion_crc8(const uint8_t *data,uint16_t len);
bool rg_sht4x_decode(const uint8_t response[6],float *temperature_c,float *humidity_rh);
#ifdef __cplusplus
}
#endif
