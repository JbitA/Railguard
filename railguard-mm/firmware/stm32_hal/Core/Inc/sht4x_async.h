#pragma once

#include "stm32h7xx_hal.h"
#include <stdbool.h>
#include <stdint.h>

typedef enum {
    SHT4X_IDLE = 0,
    SHT4X_TX,
    SHT4X_WAIT,
    SHT4X_RX
} sht4x_state_t;

typedef struct {
    I2C_HandleTypeDef *i2c;
    sht4x_state_t state;
    uint8_t command;
    uint8_t rx[6];
    uint32_t ready_at_ms;
    uint32_t next_due_ms;
    uint32_t last_valid_ms;
    float temperature_c;
    float humidity;
    bool valid;
    uint32_t errors;
} sht4x_async_t;

void sht4x_async_init(sht4x_async_t *sensor, I2C_HandleTypeDef *i2c);
void sht4x_async_service(sht4x_async_t *sensor, uint32_t now_ms);
void sht4x_async_tx_complete(sht4x_async_t *sensor, I2C_HandleTypeDef *i2c, uint32_t now_ms);
void sht4x_async_rx_complete(sht4x_async_t *sensor, I2C_HandleTypeDef *i2c, uint32_t now_ms);
void sht4x_async_error(sht4x_async_t *sensor, I2C_HandleTypeDef *i2c, uint32_t now_ms);
