#include "main.h"
#include "dsp_features.h"
#include "gnss_time.h"
#include "gnss_stream.h"
#include "iis3dwb_dma.h"
#include "pps_clock.h"
#include "railguard_packet.h"
#include "railguard_transport.h"
#include "sht4x_async.h"
#include <string.h>

extern SPI_HandleTypeDef hspi1;
extern I2C_HandleTypeDef hi2c1;
extern TIM_HandleTypeDef htim2;
extern UART_HandleTypeDef huart3;
extern PCD_HandleTypeDef hpcd_USB_OTG_FS;

static iis3dwb_bus_t sensors;
static pps_clock_t pps;
static sht4x_async_t environment;
static uint32_t sequence;
static uint8_t usb_tx[176];
static gnss_fix_t latest_fix;
static rg_gnss_stream_t gnss_stream;
static uint8_t gnss_dma_rx[256];
static volatile uint32_t watermark_epoch[IIS3DWB_SENSOR_COUNT];
static volatile uint32_t watermark_sub_us[IIS3DWB_SENSOR_COUNT];

void railguard_gnss_line(const char *line, size_t len);

static void gnss_line_callback(const char *line, size_t len, void *context) {
    (void)context;
    railguard_gnss_line(line, len);
}

static void arm_gnss_dma(void) {
    (void)HAL_UARTEx_ReceiveToIdle_DMA(&huart3, gnss_dma_rx, sizeof(gnss_dma_rx));
}

static void publish_window(void) {
    if (sensors.samples_ready == 0u) return;

    const uint8_t sensor_id = sensors.active_sensor;
    rg_vibration_features_t features;
    if (!rg_compute_vibration_features(
            sensors.samples, sensors.samples_ready, 26667.0f, &features)) {
        sensors.samples_ready = 0u;
        return;
    }

    rg_sensor_feature_payload_t payload = {0};
    payload.sensor_id = sensor_id;
    payload.window_samples = sensors.samples_ready;
    payload.sample_rate_hz = 26667.0f;
    memcpy(payload.axis_rms, features.axis_rms, sizeof(payload.axis_rms));
    payload.rms = features.rms;
    payload.peak = features.peak;
    payload.kurtosis = features.kurtosis;
    payload.crest_factor = features.crest_factor;
    memcpy(payload.band_energy, features.band_energy, sizeof(payload.band_energy));

    if (latest_fix.valid) {
        payload.flags |= RG_FLAG_GNSS_VALID;
        payload.latitude = latest_fix.latitude;
        payload.longitude = latest_fix.longitude;
        payload.speed_mps = latest_fix.speed_mps;
    }
    if (environment.valid) {
        payload.flags |= RG_FLAG_ENV_VALID;
        payload.temperature_c = environment.temperature_c;
        payload.humidity = environment.humidity;
    }

    rg_timestamp_t timestamp = {
        sequence++, watermark_epoch[sensor_id], watermark_sub_us[sensor_id]
    };
    if (timestamp.pps_epoch < 946684800u) {
        timestamp.pps_epoch = pps_clock_epoch(&pps);
        timestamp.sub_us = pps_clock_sub_us(&pps);
    }

    const size_t encoded = rg_encode_sensor_feature_packet(
        usb_tx, sizeof(usb_tx), &timestamp, &payload);
    if (encoded > 0u) {
        railguard_transport_send(usb_tx, (uint16_t)encoded);
    }
    sensors.samples_ready = 0u;
}

int main(void) {
    HAL_Init();
    SystemClock_Config();
    MX_GPIO_Init();
    MX_DMA_Init();
    MX_SPI1_Init();
    MX_I2C1_Init();
    MX_USART3_UART_Init();
    MX_TIM2_Init();
    MX_USB_OTG_FS_PCD_Init();
    railguard_transport_init();
    rg_gnss_stream_init(&gnss_stream);
    arm_gnss_dma();

    GPIO_TypeDef *ports[3] = {GPIOB, GPIOB, GPIOB};
    uint16_t pins[3] = {GPIO_PIN_0, GPIO_PIN_1, GPIO_PIN_2};
    pps_clock_init(&pps, &htim2);
    sht4x_async_init(&environment, &hi2c1);
    if (!iis3dwb_bus_init(&sensors, &hspi1, ports, pins)) {
        Error_Handler();
    }

    while (1) {
        if (!sensors.dma_busy && sensors.samples_ready > 0u) {
            publish_window();
        }
        iis3dwb_service(&sensors);
        sht4x_async_service(&environment, HAL_GetTick());
        railguard_transport_service();
    }
}

void HAL_GPIO_EXTI_Callback(uint16_t pin) {
    uint8_t sensor_id = 0xffu;
    if (pin == GPIO_PIN_0) sensor_id = 0u;
    else if (pin == GPIO_PIN_1) sensor_id = 1u;
    else if (pin == GPIO_PIN_2) sensor_id = 2u;

    if (sensor_id < IIS3DWB_SENSOR_COUNT) {
        watermark_epoch[sensor_id] = pps_clock_epoch(&pps);
        watermark_sub_us[sensor_id] = pps_clock_sub_us(&pps);
        iis3dwb_mark_watermark(&sensors, sensor_id);
    }
}

void HAL_TIM_IC_CaptureCallback(TIM_HandleTypeDef *htim) {
    if (htim->Instance == TIM2) pps_clock_on_capture(&pps);
}

void HAL_SPI_TxRxCpltCallback(SPI_HandleTypeDef *hspi) {
    iis3dwb_spi_complete(&sensors, hspi);
}

void HAL_SPI_ErrorCallback(SPI_HandleTypeDef *hspi) {
    iis3dwb_spi_error(&sensors, hspi);
}

void HAL_I2C_MasterTxCpltCallback(I2C_HandleTypeDef *hi2c) {
    sht4x_async_tx_complete(&environment, hi2c, HAL_GetTick());
}

void HAL_I2C_MasterRxCpltCallback(I2C_HandleTypeDef *hi2c) {
    sht4x_async_rx_complete(&environment, hi2c, HAL_GetTick());
}

void HAL_I2C_ErrorCallback(I2C_HandleTypeDef *hi2c) {
    sht4x_async_error(&environment, hi2c, HAL_GetTick());
}

void HAL_UARTEx_RxEventCallback(UART_HandleTypeDef *huart, uint16_t size) {
    if (huart != &huart3) return;
    if (size > sizeof(gnss_dma_rx)) size = sizeof(gnss_dma_rx);
    rg_gnss_stream_feed(&gnss_stream, gnss_dma_rx, size, gnss_line_callback, NULL);
    arm_gnss_dma();
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart) {
    if (huart == &huart3) arm_gnss_dma();
}

void railguard_gnss_line(const char *line, size_t len) {
    gnss_fix_t fix = {0};
    if (gnss_rmc_fix(line, len, &fix)) {
        latest_fix = fix;
        pps_clock_set_next_epoch(&pps, fix.epoch);
    }
}
