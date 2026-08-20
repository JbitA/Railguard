#pragma once
#include "transport_queue.h"
#include <stdbool.h>
#include <stdint.h>

void railguard_transport_init(void);
bool railguard_transport_send(const uint8_t *data, uint16_t len);
void railguard_transport_service(void);
void railguard_transport_on_tx_complete(void);
uint32_t railguard_transport_dropped(void);
uint8_t railguard_transport_depth(void);

/* Platform boundary. A target build MUST supply this symbol; there is deliberately no weak fallback. */
rg_transport_start_result_t railguard_transport_hw_start(
    const uint8_t *data, uint16_t len, void *context);
