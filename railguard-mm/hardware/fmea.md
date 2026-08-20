# Acquisition-node design FMEA

This is a design-time failure-mode review for the RailGuard-MM reference acquisition node. It is not a railway safety certification or an ISO/EN hazard analysis. Its purpose is to make expected faults, observability and mitigation explicit before hardware manufacture.

| Failure mode | System effect | Detection / observable | Design mitigation | Remaining verification |
|---|---|---|---|---|
| One IIS3DWB stops responding | Spatial fusion cannot form a valid 3-sensor tuple | Missing sensor ID / telemetry silence; no fresh aggregate | Independent CS/INT lines; fail closed rather than infer from an untrained degraded sensor set | Fault injection on each sensor, verify no false fused prediction |
| IIS3DWB FIFO overrun | Lost high-rate vibration samples | FIFO status/sequence discontinuity; packet-loss metrics | Watermark interrupt + burst DMA; bounded processing; avoid per-word DMA interrupts | Sustained multi-hour acquisition with overrun counter = 0 target |
| SPI DMA/cache incoherency | Corrupt or stale FIFO bytes on Cortex-M7 | CRC/semantic failures, impossible features | 32-byte aligned DMA buffers and explicit D-cache maintenance | Logic analyzer + injected cache-stress workload |
| GNSS PPS lost | Absolute UTC alignment degrades | PPS/clock lock state and jitter metrics | Hardware capture; runtime refuses synchronized multimodal inference until clock alignment locks | Disconnect/reconnect PPS and verify state transitions |
| GNSS serial corruption | Missing/wrong position or UTC context | NMEA checksum/parser rejection, stale validity flags | chunk-safe Receive-to-IDLE framing; context-validity flags | UART noise/truncation fault injection |
| Camera frame stall | Vision unavailable; temporal model input gap | `camera_matched=false`, sync error absent | bounded timestamp matching; temporal queues reset across gaps | unplug/stall camera during operation and verify no bridged sequence |
| V4L2 timestamp differs from true exposure instant | systematic camera↔sensor phase error despite clock alignment | sensitivity ablation / persistent sync residual; no direct exposure-edge observable yet | monotonic-domain alignment + nearest-frame bound; design reserves camera trigger capability | add 1.8 V trigger/strobe level translation and capture camera STROBE on an MCU timer for exposure-edge validation |
| Camera blur/defocus | Visual evidence becomes low-information | motion/contrast/sharpness metrics | dashboard observability; future quality gate can suppress inference if calibrated | controlled defocus experiment and threshold calibration |
| Linux clock/NTP error | Wall-clock timestamp wrong | UTC↔monotonic offset estimator remains independent of wall clock | fusion uses monotonic camera timestamps aligned from GNSS/PPS, not Linux wall time | intentionally step system clock and verify association continuity |
| USB CDC backpressure | Telemetry packets delayed/dropped | transport queue occupancy/drop counter | bounded non-blocking queue; in-flight buffer retained until transfer complete | host intentionally stalls reads; verify bounded failure behavior |
| MQTT/broker outage | Cloud telemetry delayed | outbox depth/drop count | SQLite durable producer/consumer outbox with bounded retention | disconnect broker, reconnect, verify ordered drain and drop accounting |
| NVMe/full filesystem | Event artifacts or outbox cannot grow | filesystem/write errors, spool/drop metrics | bounded telemetry outbox; event artifacts kept separate | fill-disk fault injection and recovery test |
| SHT41 missing/stuck | environment context absent/stale | CRC/read failures, validity timeout | asynchronous I²C state machine; stale validity expires | unplug/emulate CRC errors; vibration servicing must continue |
| 3.3 V rail brownout | MCU/sensors reset or misbehave | reset cause/watchdog, rail scope capture | separated sensor rail, bulk/local decoupling, watchdog | input dip/startup test across intended supply range |
| 12 V converter thermal overload | Jetson/system reset | rail droop/temperature/current | 60 W converter against 46 W design allowance; airflow/thermal margin required | thermal soak at worst intended load/environment |
| Reverse polarity/input transient | possible power-stage damage | fuse/rail failure | fuse + reverse-polarity stage; transient protection finalized during ECAD | bench transient/reverse-polarity qualification after PCB design |
| Sensor mechanical mounting changes | vibration distribution shift | cross-sensor RMS/feature distribution drift | three distinct IDs and spatial observability; fixed mounting specification required | repeatability test after remounting / torque variation |

## Fail-open versus fail-closed choices

The current multimodal inference path deliberately **fails closed** when clock alignment is unlocked, a camera interval is unmatched, or all three vibration sensors are not fresh within the skew bound. This sacrifices availability in favor of avoiding predictions from an input distribution the model was not trained to handle. A future degraded two-sensor mode should only be enabled after explicit sensor-dropout training/evaluation.
