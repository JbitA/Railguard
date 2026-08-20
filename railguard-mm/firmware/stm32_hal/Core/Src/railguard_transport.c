#include "railguard_transport.h"

static rg_transport_queue_t transport_queue;

/* railguard_transport_hw_start() is intentionally supplied by the board port.
 * There is no weak/no-op fallback: omitting the USB/UART transport adapter must
 * fail at link time rather than produce a firmware image that silently drops data.
 */

void railguard_transport_init(void) {
    rg_transport_queue_init(&transport_queue);
}

bool railguard_transport_send(const uint8_t *data, uint16_t len) {
    return rg_transport_queue_enqueue(&transport_queue, data, len);
}

void railguard_transport_service(void) {
    rg_transport_queue_service(&transport_queue, railguard_transport_hw_start, NULL);
}

void railguard_transport_on_tx_complete(void) {
    rg_transport_queue_tx_complete_isr(&transport_queue);
}

uint32_t railguard_transport_dropped(void) {
    return transport_queue.dropped;
}

uint8_t railguard_transport_depth(void) {
    return rg_transport_queue_depth(&transport_queue);
}
