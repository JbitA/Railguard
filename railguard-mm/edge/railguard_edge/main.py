from __future__ import annotations
import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import serial
from .config import load_config
from .features import vision_features
from .publisher import TelemetryPublisher
from .serial_protocol import StreamDecoder, decode_feature_payload, decode_sensor_feature_payload, validate_sensor_feature_payload, packet_timestamp_iso, packet_timestamp_ns
from .deep_inference import DeepTemporalInference
from .artifacts import ArtifactUploader


def replay(path: Path, publisher: TelemetryPublisher, realtime: bool = True):
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        publisher.publish(record)
        if realtime:
            time.sleep(0.1)


def live(config_path: str, publisher: TelemetryPublisher):
    import yaml
    raw = yaml.safe_load(Path(config_path).read_text())
    s_cfg = raw["serial"]; c_cfg = raw["camera"]
    ser = serial.Serial(s_cfg["port"], s_cfg.get("baudrate", 921600), timeout=0.05)
    cap = cv2.VideoCapture(c_cfg.get("device", "/dev/video0"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, c_cfg.get("width", 1280)); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c_cfg.get("height", 720)); cap.set(cv2.CAP_PROP_FPS, c_cfg.get("fps", 30))
    decoder = StreamDecoder(); prev_gray = None; last_seq = None; packet_loss = 0
    model_step_ms = float(raw.get("model", {}).get("sample_period_ms", 100.0))
    last_emit_ns = 0
    spatial_latest: dict[int, tuple[dict,int]] = {}; spatial_fresh: set[int] = set(); max_sensor_skew_ms=float(s_cfg.get("max_sensor_skew_ms",50.0))
    cloud_cfg=raw.get("cloud", {}); uploader=ArtifactUploader(cloud_cfg["api_base"], cloud_cfg.get("event_cache","data/event_cache")) if cloud_cfg.get("api_base") else None
    event_threshold=float(cloud_cfg.get("event_anomaly_threshold",0.85))
    infer = None
    mcfg = raw.get("model", {})
    mpath = Path(mcfg.get("path", "")) if mcfg.get("path") else None
    if mpath and mpath.exists():
        infer = DeepTemporalInference(mpath, int(mcfg.get("seq_len", 32)), mcfg.get("version", "fusion-transformer-v1"))
    try:
        while True:
            ok, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if ok else None
            vis = vision_features(gray, prev_gray) if gray is not None else {"motion_score": 0.0, "contrast": 0.0, "sharpness": 0.0}
            if gray is not None:
                prev_gray = gray
            for pkt in decoder.feed(ser.read(4096)):
                if last_seq is not None and pkt.seq > last_seq + 1:
                    packet_loss += pkt.seq - last_seq - 1
                last_seq = pkt.seq
                sensor_rows = None; sensor_skew_ms = 0.0; context_flags = 0
                if pkt.packet_type == 2:
                    if pkt.version != 2:
                        continue
                    one = decode_sensor_feature_payload(pkt.payload); validate_sensor_feature_payload(one); sid=int(one["sensor_id"])
                    spatial_latest[sid]=(one,packet_timestamp_ns(pkt)); spatial_fresh.add(sid)
                    if not {0,1,2}.issubset(spatial_fresh) or not all(i in spatial_latest for i in (0,1,2)):
                        continue
                    rows=[spatial_latest[i][0] for i in (0,1,2)]; stamps=[spatial_latest[i][1] for i in (0,1,2)]; spatial_fresh.clear()
                    sensor_skew_ms=(max(stamps)-min(stamps))/1e6 if min(stamps)>0 else 0.0
                    if sensor_skew_ms > max_sensor_skew_ms: continue
                    newest_idx=stamps.index(max(stamps)); context_flags=0
                    rms=sum(x["rms"] for x in rows)/3.0; peak=max(x["peak"] for x in rows); kurt=sum(x["kurtosis"] for x in rows)/3.0; crest=peak/(rms+1e-9); bands=[sum(x["band_energy"][j] for x in rows)/3.0 for j in range(4)]
                    valid_gnss=[i for i,x in enumerate(rows) if int(x["flags"]) & 0x01]
                    valid_env=[i for i,x in enumerate(rows) if int(x["flags"]) & 0x02]
                    gnss_idx=max(valid_gnss,key=lambda i:stamps[i]) if valid_gnss else None
                    env_idx=max(valid_env,key=lambda i:stamps[i]) if valid_env else None
                    d={"lat":None,"lon":None,"speed_mps":None,"temperature_c":None,"humidity":None}
                    if gnss_idx is not None:
                        context_flags |= 0x01; src=rows[gnss_idx]; d.update(lat=src["lat"],lon=src["lon"],speed_mps=src["speed_mps"])
                    if env_idx is not None:
                        context_flags |= 0x02; src=rows[env_idx]; d.update(temperature_c=src["temperature_c"],humidity=src["humidity"])
                    fused_ns=max(stamps)
                    if last_emit_ns and fused_ns-last_emit_ns < int(model_step_ms*1e6):
                        continue
                    last_emit_ns=fused_ns
                    sensor_rows=[{"sensor_id":x["sensor_id"],"rms_ms2":x["rms"],"peak_ms2":x["peak"],"kurtosis":x["kurtosis"],"crest_factor":x["crest_factor"],"band_energy":list(x["band_energy"])} for x in rows]
                elif pkt.packet_type == 1:
                    if pkt.version != 1:
                        continue
                    d = decode_feature_payload(pkt.payload); ax, ay, az = d["axis_rms"]; rms = math.sqrt((ax*ax + ay*ay + az*az) / 3.0); peak = max(ax, ay, az) * 3.0; kurt=3.0; crest=peak/(rms+1e-9); bands=[0.0]*4
                else:
                    continue
                ts = packet_timestamp_iso(pkt, datetime.now(timezone.utc)); vibration={"rms_ms2":rms,"peak_ms2":peak,"kurtosis":kurt,"crest_factor":crest,"band_energy":bands}
                if sensor_rows is not None: vibration["sensors"]=sensor_rows
                record = {
                    "schema_version": 1, "device_id": raw["device_id"], "ts": ts, "seq": pkt.seq, "sample_period_ms": model_step_ms,
                    "gps": {"lat": d["lat"], "lon": d["lon"], "speed_mps": d["speed_mps"]},
                    "environment": {"temperature_c": d["temperature_c"], "humidity": d.get("humidity")},
                    "vibration": vibration,
                    "vision": {"motion_score": vis["motion_score"], "contrast": vis["contrast"], "sharpness": vis["sharpness"], "frame_ref": None},
                    "health": {"packet_loss": packet_loss, "spool_depth": publisher.spool_depth(), "sensor_skew_ms": sensor_skew_ms, "context_flags": context_flags}
                }
                # The public training table contains complete GNSS/environment context.
                # Until explicit missingness/dropout training is added, fail closed instead
                # of feeding physical zeroes for absent context into the deployed model.
                context_complete = (context_flags & 0x03) == 0x03
                if infer is not None and ok and context_complete:
                    pred = infer.update(frame, record)
                    if pred is not None:
                        record["prediction"] = pred
                        if uploader is not None and float(pred.get("anomaly_probability",0.0)) >= event_threshold:
                            ref=uploader.upload_frame(raw["device_id"],ts,frame)
                            if ref: record["vision"]["frame_ref"]=ref
                if uploader is not None and pkt.seq % 100 == 0:
                    uploader.flush(raw["device_id"], limit=2)
                publisher.publish(record)
    finally:
        cap.release(); ser.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--replay", type=Path)
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    pub = TelemetryPublisher(cfg.mqtt, cfg.spool_path); pub.start()
    try:
        replay(args.replay, pub, realtime=not args.fast) if args.replay else live(args.config, pub)
    finally:
        pub.close()

if __name__ == "__main__":
    main()
