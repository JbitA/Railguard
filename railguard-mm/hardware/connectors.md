# Connector and harness definition

| Ref | Interface | Minimum implementation | Purpose |
|---|---|---|---|
| J1 | 2-pin locking power | >=5 A contact rating, keyed | 9-36 V vehicle input |
| J2 | retained USB 3 | shielded SuperSpeed cable | Jetson to global-shutter camera |
| J3 | USB 2 FS internal | D+/D-/GND plus shield/retention | STM32 CDC ACM to Jetson |
| J4 | ARM Cortex 10-pin 1.27 mm | SWDIO/SWCLK/SWO/NRST/3V3/GND | programming and trace |
| J5 | GNSS antenna | 50 Ω RF connector matching carrier | active multi-band antenna |
| TP1 | test point | +12V_SYS | system-rail bring-up |
| TP2 | test point | +3V3_SENS | acquisition-rail bring-up |
| TP3 | test point | GNSS_PPS | timing validation |
| TP4 | test point | SPI1_SCK | signal-integrity validation |

For a mechanically distributed sensor installation, do **not** extend 10 MHz single-ended SPI through long vehicle harnesses. Either keep each IIS3DWB close to the acquisition PCB/short flex, or convert a remote sensor head to a robust differential link (CAN-FD/RS-485/Ethernet) with local sampling and timestamping.
