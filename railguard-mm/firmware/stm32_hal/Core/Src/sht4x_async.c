#include "sht4x_async.h"
#include "environment_decode.h"
#include <string.h>

#define SHT4X_I2C_ADDRESS      (0x44u << 1u)
#define SHT4X_HIGH_PRECISION   0xFDu
#define SHT4X_CONVERSION_MS    10u
#define SHT4X_SAMPLE_PERIOD_MS 1000u
#define SHT4X_STALE_AFTER_MS   5000u

static bool deadline_reached(uint32_t now_ms, uint32_t deadline_ms) {
    return (int32_t)(now_ms - deadline_ms) >= 0;
}

static void schedule_retry(sht4x_async_t *sensor, uint32_t now_ms) {
    sensor->state = SHT4X_IDLE;
    sensor->next_due_ms = now_ms + SHT4X_SAMPLE_PERIOD_MS;
}

void sht4x_async_init(sht4x_async_t *sensor, I2C_HandleTypeDef *i2c) {
    memset(sensor, 0, sizeof(*sensor));
    sensor->i2c = i2c;
    sensor->command = SHT4X_HIGH_PRECISION;
    sensor->next_due_ms = 100u;
}

void sht4x_async_service(sht4x_async_t *sensor, uint32_t now_ms) {
    if (sensor == NULL || sensor->i2c == NULL) return;

    if (sensor->valid && (uint32_t)(now_ms - sensor->last_valid_ms) > SHT4X_STALE_AFTER_MS) {
        sensor->valid = false;
    }

    if (sensor->state == SHT4X_IDLE && deadline_reached(now_ms, sensor->next_due_ms)) {
        if (HAL_I2C_Master_Transmit_IT(sensor->i2c, SHT4X_I2C_ADDRESS, &sensor->command, 1u) == HAL_OK) {
            sensor->state = SHT4X_TX;
        } else {
            ++sensor->errors;
            schedule_retry(sensor, now_ms);
        }
        return;
    }

    if (sensor->state == SHT4X_WAIT && deadline_reached(now_ms, sensor->ready_at_ms)) {
        if (HAL_I2C_Master_Receive_IT(sensor->i2c, SHT4X_I2C_ADDRESS, sensor->rx, sizeof(sensor->rx)) == HAL_OK) {
            sensor->state = SHT4X_RX;
        } else {
            ++sensor->errors;
            schedule_retry(sensor, now_ms);
        }
    }
}

void sht4x_async_tx_complete(sht4x_async_t *sensor, I2C_HandleTypeDef *i2c, uint32_t now_ms) {
    if (sensor == NULL || i2c != sensor->i2c || sensor->state != SHT4X_TX) return;
    sensor->ready_at_ms = now_ms + SHT4X_CONVERSION_MS;
    sensor->state = SHT4X_WAIT;
}

void sht4x_async_rx_complete(sht4x_async_t *sensor, I2C_HandleTypeDef *i2c, uint32_t now_ms) {
    if (sensor == NULL || i2c != sensor->i2c || sensor->state != SHT4X_RX) return;

    float temperature_c = 0.0f;
    float humidity = 0.0f;
    if (rg_sht4x_decode(sensor->rx, &temperature_c, &humidity)) {
        sensor->temperature_c = temperature_c;
        sensor->humidity = humidity;
        sensor->last_valid_ms = now_ms;
        sensor->valid = true;
    } else {
        ++sensor->errors;
    }
    schedule_retry(sensor, now_ms);
}

void sht4x_async_error(sht4x_async_t *sensor, I2C_HandleTypeDef *i2c, uint32_t now_ms) {
    if (sensor == NULL || i2c != sensor->i2c) return;
    ++sensor->errors;
    schedule_retry(sensor, now_ms);
}
