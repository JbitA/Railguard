# STM32H743 production acquisition path

This directory is the hardware-oriented implementation for the custom STM32H743VI acquisition board. It is intentionally separate from the PlatformIO bench scaffold because CubeMX/HAL owns clocks, DMA routing, USB device middleware and interrupt-vector generation.

## CubeMX configuration contract

- **SPI1**: master, 8-bit, **mode 0**, <=10 MHz; TX/RX DMA enabled. PA5/PA6/PA7.
- **CS**: PB0/PB1/PB2 push-pull outputs, default high.
- **FIFO watermark EXTI**: PC0/PC1/PC2, rising edge. The IIS3DWB INT1 pin must be held low or unconnected during sensor power-up; board design provides the required bias behavior.
- **TIM2 CH1 / PA0**: 1 MHz counter; input capture from GNSS PPS. Configure slave reset mode with TI1FP1 as trigger so CNT resets in hardware on the PPS edge; the ISR only associates the UTC epoch.
- **USART3**: GNSS input using circular DMA/IDLE-line handling. Pass complete RMC sentences to `railguard_gnss_line()`.
- **I2C1**: PB8/PB9 with 4.7 kΩ pull-ups to 3.3 V; SHT41 at address 0x44. The supplied state machine uses interrupt-driven transactions and CRC-checks both returned words.
- **USB FS CDC**: compile `Integration/usb_cdc_port.c` with the generated CDC middleware and call `railguard_transport_on_tx_complete()` from `CDC_TransmitCplt_FS`. The application queue has no weak/no-op hardware fallback, so omitting the adapter fails at link time instead of silently dropping telemetry. UART DMA can be used during bring-up with an equivalent target adapter.
- **D-cache**: place SPI DMA buffers in a non-cacheable MPU region or perform explicit cache clean/invalidate around DMA transfers on STM32H7.

## Acquisition state machine

1. Configure all three IIS3DWB devices for ±16 g, FIFO batching at 26.667 kHz, 512-word watermark and FIFO watermark interrupt.
2. EXTI callbacks only set a pending bit; they do not perform SPI work.
3. The main service selects a pending sensor and reads FIFO status.
4. FIFO words are drained in one watermark-sized `HAL_SPI_TransmitReceive_DMA` burst. The device automatically wraps from `FIFO_DATA_OUT_Z_H` back to `FIFO_DATA_OUT_TAG`, so one DMA completion can contain many tagged XYZ words.
5. Acceleration is converted using the ±16 g sensitivity (0.488 mg/LSB), then RMS, peak, kurtosis, crest-factor and four Goertzel-band features are generated without heap allocation.
6. A non-blocking SHT41 state machine samples temperature/humidity at low rate without delaying FIFO service; CRC-valid context is attached to sensor packets.
7. The packet uses the PPS-associated UTC second plus the hardware timer's microsecond offset and is framed with sensor identity, sequence number + CRC32.

The code deliberately separates the CubeMX-generated board initialization from application logic so the DMA acquisition, timestamp discipline and packet format remain version-controlled and reviewable.
