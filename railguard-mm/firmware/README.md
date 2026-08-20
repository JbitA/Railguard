# STM32 acquisition firmware

Two targets are kept for different purposes:

- `src/main.cpp` is the lightweight PlatformIO/Nucleo protocol bring-up target.
- `stm32_hal/` is the hardware-oriented STM32H743 implementation used by the system design. It contains IIS3DWB FIFO configuration, watermark EXTI handling, SPI DMA draining, GNSS/PPS epoch discipline, static feature buffers and CRC-framed transport.

For the custom STM32H743VI board, generate the clock/peripheral startup with STM32CubeMX using `stm32_hal/README.md`, then add the version-controlled files in `stm32_hal/Core` to the project. This keeps generated HAL startup code out of the repository while retaining the acquisition logic that determines real-time behavior.
