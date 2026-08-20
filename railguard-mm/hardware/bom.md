# Reference BOM

| Ref | Qty | Part | Function | Selection rationale |
|---|---:|---|---|---|
| U1 | 1 | NVIDIA Jetson Orin Nano Super Developer Kit / production Orin Nano module + carrier | Edge AI compute | CUDA/TensorRT, high AI throughput and native Linux networking/vision stack. |
| U2 | 1 | STM32H743VIT6 | Real-time acquisition MCU | 480 MHz Cortex-M7, DSP/FPU, multiple SPI/UART, USB and large SRAM. |
| U3-U5 | 3 | IIS3DWBTR | Triaxial vibration sensors | 26.7 kHz ODR, wide-band industrial vibration measurement, FIFO, digital SPI. |
| CAM1 | 1 | See3CAM_50CUGM | Global-shutter machine-vision camera | 5 MP Sony Pregius IMX264, USB 3 UVC/Linux, trigger support. |
| U6 | 1 | ZED-F9P module or carrier | GNSS/RTK + timing | Multi-band GNSS, PPS and precise spatial indexing. |
| SSD1 | 1 | 256 GB NVMe M.2 | Local ring buffer | Raw event retention and network-outage spool. |
| U7 | 1 | MEAN WELL RSDW60F-12 | 24 V-class to 12 V isolated DC/DC | 9-36 V input, 12 V / 5 A, 60 W; appropriate power class for the Jetson DC input. |
| U8 | 1 | TPS62130-class synchronous buck | 12 V to 3.3 V acquisition rail | Dedicated efficient logic/sensor power with >1 A margin. |
| U9 | 1 | Sensirion SHT41 | Ambient temperature / relative humidity | Digital I²C environmental context with CRC-protected measurements; physically place away from power/compute heat sources. |
| F1 | 1 | 5 A time-delay fuse | Input protection | Limits fault energy ahead of the DC/DC module. |
| Q1 | 1 | Reverse-polarity MOSFET/ideal-diode stage | Input protection | Prevents damage from swapped supply polarity. |
| J1 | 1 | Locking 2-pin power connector | Vehicle input | Vibration-resistant service connection. |
| J2 | 1 | USB 3 retained cable/connector | Camera | SuperSpeed UVC path. |
| J3 | 1 | USB 2 retained cable/header | STM32 to Jetson | CRC-framed acquisition data. |
| J4 | 1 | 10-pin Cortex SWD header | Debug/programming | Direct STM32 recovery and trace access. |
| ANT1 | 1 | Active multi-band GNSS antenna | GNSS | Reliable sky-facing reception. |
| Cx | as req. | X7R + electrolytic capacitors | Decoupling/bulk | Local high-frequency bypass and rail stability. |
| Rx | as req. | 10 kΩ CS pulls, 22-33 Ω series footprints | Digital conditioning | Defined startup state and edge-rate tuning. |

## Notes

The reference design deliberately uses a qualified DC/DC module for the high-current rail instead of turning the power converter into the main project risk. The custom electronics focus on deterministic sensing, timing and edge integration. Production work still requires derating, thermal/EMC validation, enclosure design and the applicable railway standards.
