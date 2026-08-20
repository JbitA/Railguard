/*
 * Compile this file inside the CubeMX STM32H743 target after enabling USB Device
 * CDC. It supplies the strong transport-start implementation used by the
 * version-controlled application queue.
 */
#include "railguard_transport.h"
#include "usbd_def.h"
#include "usbd_cdc_if.h"

rg_transport_start_result_t railguard_transport_hw_start(
    const uint8_t *data, uint16_t len, void *context) {
    (void)context;
    const uint8_t status = CDC_Transmit_FS((uint8_t *)data, len);
    if (status == USBD_OK) return RG_TRANSPORT_STARTED;
    if (status == USBD_BUSY) return RG_TRANSPORT_BUSY;
    return RG_TRANSPORT_ERROR;
}

/*
 * In the CubeMX-generated CDC_TransmitCplt_FS callback, add:
 *
 *     railguard_transport_on_tx_complete();
 *
 * The queue retains the in-flight slot until that callback fires, so the USB
 * middleware never observes a buffer that has been overwritten by the producer.
 */
