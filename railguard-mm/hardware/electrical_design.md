# Electrical reference design

This file defines the practical carrier/acquisition-board connections needed to turn the architecture into hardware.

## A. Power entry and distribution

### J1 — vehicle input

- Pin 1: `VIN_9V_36V`
- Pin 2: `GND_CHASSIS_RETURN`
- F1: 5 A time-delay fuse in series with Pin 1
- Reverse-polarity stage: P-channel MOSFET or ideal-diode controller rated above the maximum input/transient envelope
- Bulk at DC/DC input: 100 µF electrolytic + 1 µF X7R + 100 nF X7R close to the module pins

### U7 — RSDW60F-12

- Input: 9-36 V nominal range
- Output: `+12V_SYS`, 5 A max
- Output bulk near Jetson connector: 470 µF low-ESR + 10 µF + 100 nF
- `+12V_SYS` branches to the Jetson barrel/DC connector and acquisition-board 3.3 V buck

### U8 — 3.3 V synchronous buck

Reference class: TPS62130 or equivalent 12 V-capable, >=1 A synchronous converter.

- Input: `+12V_SYS`
- Output: `+3V3_SENS`
- Target load: MCU + 3 MEMS + GNSS + margin < 1 A
- Output: 22 µF + 1 µF + 100 nF at regulator, then local decoupling at each load

## B. STM32H743 core

### Supplies

- All VDD pins -> `+3V3_SENS`
- Each VDD pin -> 100 nF to ground, placed adjacent
- Per supply group -> 4.7 µF bulk
- VDDA/VREF+ -> `+3V3_SENS` through ferrite bead footprint; 1 µF + 100 nF to ground
- VCAP1, VCAP2 -> 2.2 µF low-ESR to ground
- VBAT -> `+3V3_SENS` for reference build, optional backup battery footprint

### Reset/debug

- NRST -> 10 kΩ to 3.3 V, 100 nF to ground, SWD connector
- BOOT0 -> 100 kΩ to ground, accessible test pad
- 10-pin Cortex header: 3V3, GND, SWDIO, SWCLK, SWO, NRST

### USB FS to Jetson

- USB_DP / USB_DM -> 22 Ω series footprints -> ESD array -> USB connector
- Route D+/D- as a short 90 Ω differential pair
- VBUS used only for USB presence sensing unless the board is explicitly configured to be bus powered
- Shield termination follows enclosure/chassis EMC design

## C. IIS3DWB sensors

All three sensors share SPI1 clock/data and have independent chip selects/interrupts.

| Net | Sensor 1 | Sensor 2 | Sensor 3 |
|---|---|---|---|
| SCK | shared | shared | shared |
| MOSI | shared | shared | shared |
| MISO | shared | shared | shared |
| CS | `ACC1_CS` | `ACC2_CS` | `ACC3_CS` |
| INT1 | `ACC1_INT` | `ACC2_INT` | `ACC3_INT` |

Per sensor:

- VDD/VDDIO -> `+3V3_SENS`
- 100 nF + 1 µF decoupling
- CS -> 10 kΩ pull-up to 3.3 V
- dedicated mechanical mounting location reference in enclosure drawing
- no copper keep-out is required by principle, but avoid high-current/switch-node routing around the package

## D. GNSS

### ZED-F9P interface

- module supply -> `+3V3_SENS` or vendor-recommended supply variant
- UART TX -> STM32 USART RX
- UART RX <- STM32 USART TX
- TIMEPULSE/PPS -> STM32 timer input capture
- RF input -> active multi-band antenna connector according to selected ZED-F9P carrier/module design
- antenna connector placed at enclosure edge with short controlled RF path


## E. Environmental context — SHT41

The reference acquisition board uses a Sensirion SHT41 on I2C1 so temperature and relative humidity are measured rather than represented by a firmware placeholder.

- VDD -> `+3V3_SENS`; 100 nF X7R placed at the device
- SCL -> STM32 PB8 / I2C1_SCL
- SDA -> STM32 PB9 / I2C1_SDA
- 4.7 kΩ pull-ups from SCL and SDA to `+3V3_SENS`
- place the sensor near a ventilated board edge and away from the 12 V DC/DC, 3.3 V buck, Jetson exhaust and other self-heating components
- firmware uses interrupt-driven I2C transactions; the measurement conversion delay is handled as a state-machine wait rather than blocking vibration FIFO servicing
- each returned temperature/humidity word is accepted only after its Sensirion CRC-8 passes

Environmental measurements are context features, not safety-critical measurements; enclosure airflow and thermal offset must be characterized if absolute ambient accuracy matters.

## F. Camera and NVMe

- See3CAM_50CUGM -> Jetson USB 3 Type-A using a retained/locking cable strategy
- camera receives 5 V over USB from the Jetson carrier
- NVMe SSD -> Jetson M.2 Key-M slot

## G. Harness

Recommended labeled harnesses:

1. `PWR-IN`: vehicle supply to J1
2. `CAM-USB3`: Jetson to camera
3. `GNSS-RF`: GNSS module to roof/sky antenna
4. `DEBUG-USB`: STM32 USB FS to Jetson internal USB port/header adapter
5. `SENSOR-MOUNT`: if MEMS devices are remote daughterboards, use shielded differential/serialized links rather than long raw SPI; the reference PCB assumes the IIS3DWB devices are physically on the acquisition board or within a short flex/cable distance.

## H. PCB stack and placement

Reference four-layer stack:

- L1: signals/components
- L2: solid ground
- L3: power + slower signals
- L4: signals/components

Place the acquisition MCU and vibration sensors away from the DC/DC module and Jetson fan/magnetic components. Place the SHT41 at a ventilated edge, thermally isolated from those heat sources. Put the GNSS RF section at the opposite edge from digital clocks. Maintain test points for 12 V, 3.3 V, NRST, PPS, SPI SCK and each sensor interrupt.
