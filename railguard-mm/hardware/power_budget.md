# Power budget

| Load | Supply path | Design allowance |
|---|---|---:|
| Jetson Orin Nano Super module/carrier | 12 V input -> carrier regulators | 30 W system allowance |
| NVMe SSD | Jetson carrier | 5 W |
| See3CAM_50CUGM | Jetson USB 5 V | 1.5 W |
| STM32 + 3 x IIS3DWB | 3.3 V acquisition rail | 1.0 W |
| GNSS + active antenna | 3.3 V acquisition rail | 1.0 W |
| SHT41 environmental sensor + I²C pull-ups | 3.3 V acquisition rail | <0.01 W typical measurement duty; included in margin |
| Conversion / startup margin | — | 7.5 W |
| **Total design allowance** | — | **46 W** |

The RSDW60F-12 is rated 60 W (12 V, 5 A), giving roughly 14 W margin over this conservative 46 W allowance.

At 24 V input and approximately 90% conversion efficiency, 46 W delivered corresponds to about 2.13 A input current. A 5 A time-delay fuse provides startup/transient margin while still limiting fault energy.

The acquisition rail is separately generated from 12 V so sensor/MCU operation does not depend on USB VBUS availability.
