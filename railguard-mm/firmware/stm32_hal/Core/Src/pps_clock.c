#include "pps_clock.h"
void pps_clock_init(pps_clock_t *c,TIM_HandleTypeDef *timer){c->timer=timer;c->epoch=0;c->next_epoch=0;c->next_valid=false;c->captures=0;HAL_TIM_IC_Start_IT(timer,TIM_CHANNEL_1);}
void pps_clock_set_next_epoch(pps_clock_t *c,uint32_t unix_epoch){c->next_epoch=unix_epoch;c->next_valid=true;}
void pps_clock_on_capture(pps_clock_t *c){
 /* TIM2 is configured in reset-slave mode from TI1FP1, so the hardware edge resets CNT independent of ISR latency. */
 c->captures++; if(c->next_valid){c->epoch=c->next_epoch;c->next_valid=false;}else if(c->epoch>=946684800u){c->epoch++;}
}
uint32_t pps_clock_epoch(const pps_clock_t *c){return c->epoch;}
uint32_t pps_clock_sub_us(const pps_clock_t *c){return __HAL_TIM_GET_COUNTER(c->timer);}
