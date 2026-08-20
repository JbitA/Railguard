#pragma once
#include "stm32h7xx_hal.h"
#include <stdbool.h>
#include <stdint.h>
#include "dsp_features.h"
#define IIS3DWB_SENSOR_COUNT 3u
#define IIS3DWB_FIFO_WORD_BYTES 7u
#define IIS3DWB_MAX_DRAIN_WORDS 512u
#define IIS3DWB_DMA_BYTES (1u + IIS3DWB_MAX_DRAIN_WORDS * IIS3DWB_FIFO_WORD_BYTES)
typedef rg_xyz_i16_t iis3dwb_xyz_t;
typedef struct {
 SPI_HandleTypeDef *spi; GPIO_TypeDef *cs_port[IIS3DWB_SENSOR_COUNT]; uint16_t cs_pin[IIS3DWB_SENSOR_COUNT];
 volatile uint8_t pending_mask; volatile bool dma_busy; uint8_t active_sensor; uint16_t transfer_words; uint16_t samples_ready;
 _Alignas(32) uint8_t tx[IIS3DWB_DMA_BYTES]; _Alignas(32) uint8_t rx[IIS3DWB_DMA_BYTES];
 iis3dwb_xyz_t samples[IIS3DWB_MAX_DRAIN_WORDS]; uint32_t fifo_overruns; uint32_t dma_errors;
} iis3dwb_bus_t;
bool iis3dwb_bus_init(iis3dwb_bus_t *b,SPI_HandleTypeDef *spi,GPIO_TypeDef **ports,const uint16_t *pins);
void iis3dwb_mark_watermark(iis3dwb_bus_t *b,uint8_t sensor);
void iis3dwb_service(iis3dwb_bus_t *b);
void iis3dwb_spi_complete(iis3dwb_bus_t *b,SPI_HandleTypeDef *spi);
void iis3dwb_spi_error(iis3dwb_bus_t *b,SPI_HandleTypeDef *spi);
float iis3dwb_rms_ms2(const iis3dwb_xyz_t *samples,uint16_t count,unsigned axis);
