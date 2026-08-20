#pragma once
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define RG_TRANSPORT_QUEUE_DEPTH 8u
#define RG_TRANSPORT_MAX_PACKET 176u

typedef enum {
    RG_TRANSPORT_STARTED = 0,
    RG_TRANSPORT_BUSY = 1,
    RG_TRANSPORT_ERROR = 2,
} rg_transport_start_result_t;

typedef rg_transport_start_result_t (*rg_transport_start_fn)(
    const uint8_t *data, uint16_t len, void *context);

typedef struct {
    uint8_t slots[RG_TRANSPORT_QUEUE_DEPTH][RG_TRANSPORT_MAX_PACKET];
    uint16_t lengths[RG_TRANSPORT_QUEUE_DEPTH];
    uint8_t head;
    uint8_t tail;
    uint8_t count;
    bool in_flight;
    volatile bool tx_complete;
    uint32_t enqueued;
    uint32_t completed;
    uint32_t dropped;
    uint32_t busy_retries;
    uint32_t start_errors;
} rg_transport_queue_t;

void rg_transport_queue_init(rg_transport_queue_t *queue);
bool rg_transport_queue_enqueue(rg_transport_queue_t *queue, const uint8_t *data, uint16_t len);
void rg_transport_queue_service(rg_transport_queue_t *queue, rg_transport_start_fn start, void *context);
void rg_transport_queue_tx_complete_isr(rg_transport_queue_t *queue);
uint8_t rg_transport_queue_depth(const rg_transport_queue_t *queue);
