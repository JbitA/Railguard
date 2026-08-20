# Hardware bring-up and verification plan

All values below are **engineering acceptance targets for future bench validation**. They are not measured results in the current repository.

## Stage 1 — unpowered inspection

1. Check polarity/orientation of MCU, IIS3DWB devices, SHT41, GNSS and power modules.
2. Measure resistance from `+12V_SYS` and `+3V3_SENS` to ground before applying power.
3. Verify SWD, PPS, SPI clock, sensor interrupts and both power rails are accessible at test points.
4. Inspect USB D+/D- continuity and isolation from power nets.

Pass criterion: no assembly short/open identified by inspection/continuity checks.

## Stage 2 — power tree only

Use a current-limited bench supply before connecting Jetson or USB.

| Test | Acceptance target |
|---|---|
| Input startup | clean startup throughout intended 9–36 V operating range |
| `+12V_SYS` steady state | within downstream Jetson/carrier input tolerance under representative load |
| `+3V3_SENS` steady state | 3.3 V ±3% at MCU and farthest sensor |
| `+3V3_SENS` ripple | <50 mVpp target under acquisition activity |
| abnormal current | no unexplained current rise or thermal hotspot |

Capture startup and shutdown waveforms rather than recording only multimeter values.

## Stage 3 — MCU/debug

1. Program via SWD and record reset-cause registers.
2. Exercise watchdog recovery.
3. Verify TIM2 free-running timing and PPS input-capture interrupt.
4. Run the portable host-test vectors on target where practical.

Pass target: stable boot/reboot behavior with no unexpected watchdog/reset cause during a one-hour idle run.

## Stage 4 — each vibration sensor independently

For sensor IDs 0, 1 and 2:

1. Read device identity/configuration registers.
2. Verify 26.667 kHz configured ODR and FIFO watermark operation.
3. Observe INT1 and SPI burst transaction on a logic analyzer.
4. Confirm each FIFO service reads a multiple of the 7-byte tagged FIFO word.
5. Apply a repeatable vibration/tap and confirm the intended sensor ID responds.

Pass target: correct identity and configuration, stable watermark cadence, valid tagged data and no cross-wired sensor ID.

## Stage 5 — simultaneous three-sensor stress

Run all three IIS3DWB devices continuously with camera/GNSS/environment traffic active.

Nominal feature transport is roughly 3 × (26,667 / 512) ≈ **156 sensor-feature packets/s**, before higher-level resampling. At 94 bytes per packet this is only about **14.7 kB/s of binary payload**, so the transport has substantial nominal bandwidth margin; the stress test should nevertheless exercise burstiness and host stalls.

Acceptance targets:

- zero sensor FIFO overruns in a one-hour nominal run;
- sequence discontinuities = 0 without intentional fault injection;
- DMA completion/watermark service remains bounded during SHT41 and GNSS traffic;
- sensor-to-sensor feature timestamp skew remains within the configured fusion limit.

## Stage 6 — GNSS/PPS timing

1. Feed valid RMC/NMEA and PPS.
2. Verify PPS capture and UTC association.
3. Compare embedded timestamps with a logic-analyzer PPS reference.
4. Disconnect PPS while serial NMEA continues, then restore it.
5. Intentionally step Linux wall clock on the Jetson and confirm camera/sensor matching uses monotonic alignment rather than wall time.

Pass target: lock/unlock state transitions are observable and multimodal inference remains disabled until alignment is valid.

## Stage 7 — USB CDC and backpressure

1. Sustain nominal telemetry while Jetson consumes continuously.
2. Stall host reads to fill the bounded CDC queue.
3. Resume host reads.
4. Verify no in-flight buffer is overwritten before transfer completion.

Pass target: queue overflow is bounded/observable, firmware remains responsive and normal transmission resumes without reset.

## Stage 8 — camera and multimodal association

1. Capture V4L2 kernel monotonic timestamps.
2. Verify nearest-frame selection against PPS-aligned sensor timestamps.
3. Introduce controlled camera stalls and latency.
4. Defocus the camera and record sharpness/contrast response.

Pass target: missing/out-of-window frames are rejected, temporal queues reset over gaps, and camera-quality degradation is visible in telemetry.

## Stage 9 — power/thermal soak

Exercise Jetson inference, camera, NVMe and acquisition simultaneously at intended enclosure airflow.

Record:

- input voltage/current and calculated power;
- 12 V and 3.3 V rail ripple;
- DC/DC, buck, Jetson and enclosure temperatures;
- resets, FIFO overruns, packet loss and outbox growth.

The repository's 46 W figure is a **design allowance**, not a measured thermal result. Final acceptance requires margin at the intended ambient temperature and enclosure configuration.

## Stage 10 — fault-injection regression

Repeat relevant FMEA faults: one missing IIS3DWB, bad sensor values, GNSS corruption/PPS loss, camera unplug, broker outage, disk-full condition and environmental-sensor CRC errors. Save logs/plots as release evidence under a future `artifacts/hardware_validation/` directory.
