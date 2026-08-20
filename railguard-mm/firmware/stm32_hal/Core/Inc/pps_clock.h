#pragma once
#include "stm32h7xx_hal.h"
#include <stdbool.h>
#include <stdint.h>
typedef struct { TIM_HandleTypeDef *timer; volatile uint32_t epoch; volatile uint32_t next_epoch; volatile bool next_valid; volatile uint32_t captures; } pps_clock_t;
void pps_clock_init(pps_clock_t *c,TIM_HandleTypeDef *timer);
void pps_clock_set_next_epoch(pps_clock_t *c,uint32_t unix_epoch);
void pps_clock_on_capture(pps_clock_t *c);
uint32_t pps_clock_epoch(const pps_clock_t *c);
uint32_t pps_clock_sub_us(const pps_clock_t *c);
