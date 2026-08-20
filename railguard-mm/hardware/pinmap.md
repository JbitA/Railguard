# STM32H743VIT6 reference pin map

The acquisition-board reference uses the **STM32H743VIT6 (LQFP100)**. The mapping below is concrete at the GPIO/alternate-function level; verify package pin numbers against the selected STM32H743VI symbol/footprint during ECAD capture.

| Function | STM32 signal | GPIO | Direction | Notes |
|---|---|---|---|---|
| IIS3DWB shared SCK | SPI1_SCK | PA5 | out | start at 10 MHz; footprint for source termination |
| IIS3DWB shared MISO | SPI1_MISO | PA6 | in | sensors tri-state when CS high |
| IIS3DWB shared MOSI | SPI1_MOSI | PA7 | out | configuration writes |
| Sensor 1 CS | GPIO | PB0 | out | active low, 10 kΩ pull-up |
| Sensor 2 CS | GPIO | PB1 | out | active low, 10 kΩ pull-up |
| Sensor 3 CS | GPIO | PB2 | out | active low, 10 kΩ pull-up |
| Sensor 1 INT1 | EXTI | PC0 | in | FIFO watermark |
| Sensor 2 INT1 | EXTI | PC1 | in | FIFO watermark |
| Sensor 3 INT1 | EXTI | PC2 | in | FIFO watermark |
| GNSS TX -> MCU | USART3_RX | PB11 | in | UBX/NMEA |
| GNSS RX <- MCU | USART3_TX | PB10 | out | configuration/corrections |
| GNSS PPS | TIM2_CH1 | PA0 | input capture | UTC epoch discipline |
| SHT41 SCL | I2C1_SCL | PB8 | bi-dir | 4.7 kΩ pull-up to 3.3 V |
| SHT41 SDA | I2C1_SDA | PB9 | bi-dir | 4.7 kΩ pull-up to 3.3 V |
| Jetson USB data | USB_OTG_FS_DM | PA11 | bi-dir | CDC ACM D- |
| Jetson USB data | USB_OTG_FS_DP | PA12 | bi-dir | CDC ACM D+ |
| Camera trigger | GPIO | PD2 | out | optional isolated/level-conditioned output |
| Status LED | GPIO | PB14 | out | heartbeat/fault code |
| Edge reset request | GPIO | PC13 | open-drain out | optional supervisor function |
| SWDIO | SWDIO | PA13 | bi-dir | debug/programming |
| SWCLK | SWCLK | PA14 | in | debug/programming |
| SWO | TRACESWO | PB3 | out | optional trace |
| NRST | NRST | NRST | in | reset/header |

## Packet framing

```text
SYNC(2) | VERSION(1) | TYPE(1) | LENGTH(2) | SEQ(4) | PPS_EPOCH(4)
SUB_US(4) | PAYLOAD(N) | CRC32(4)
```

`PPS_EPOCH` is Unix seconds associated with the latest valid GNSS PPS. `SUB_US` is the hardware-timer offset from that PPS. During bench bring-up, an uninitialized epoch is explicitly detected by the Jetson and Linux receive time is used as a fallback.
