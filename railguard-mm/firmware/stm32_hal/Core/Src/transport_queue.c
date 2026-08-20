#include "transport_queue.h"
#include <string.h>

void rg_transport_queue_init(rg_transport_queue_t *queue) {
    if (queue == NULL) return;
    memset(queue, 0, sizeof(*queue));
}

bool rg_transport_queue_enqueue(rg_transport_queue_t *queue, const uint8_t *data, uint16_t len) {
    if (queue == NULL || data == NULL || len == 0u || len > RG_TRANSPORT_MAX_PACKET) return false;
    if (queue->count >= RG_TRANSPORT_QUEUE_DEPTH) {
        queue->dropped++;
        return false;
    }

    memcpy(queue->slots[queue->tail], data, len);
    queue->lengths[queue->tail] = len;
    queue->tail = (uint8_t)((queue->tail + 1u) % RG_TRANSPORT_QUEUE_DEPTH);
    queue->count++;
    queue->enqueued++;
    return true;
}

void rg_transport_queue_tx_complete_isr(rg_transport_queue_t *queue) {
    if (queue == NULL || !queue->in_flight) return;
    queue->tx_complete = true;
}

void rg_transport_queue_service(rg_transport_queue_t *queue, rg_transport_start_fn start, void *context) {
    if (queue == NULL || start == NULL) return;

    if (queue->in_flight && queue->tx_complete) {
        queue->tx_complete = false;
        queue->in_flight = false;
        if (queue->count > 0u) {
            queue->head = (uint8_t)((queue->head + 1u) % RG_TRANSPORT_QUEUE_DEPTH);
            queue->count--;
            queue->completed++;
        }
    }

    if (queue->in_flight || queue->count == 0u) return;

    queue->tx_complete = false;
    queue->in_flight = true;
    const rg_transport_start_result_t result = start(
        queue->slots[queue->head], queue->lengths[queue->head], context);

    if (result == RG_TRANSPORT_STARTED) return;

    queue->in_flight = false;
    if (result == RG_TRANSPORT_BUSY) queue->busy_retries++;
    else queue->start_errors++;
}

uint8_t rg_transport_queue_depth(const rg_transport_queue_t *queue) {
    return queue == NULL ? 0u : queue->count;
}
