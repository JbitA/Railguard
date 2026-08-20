from __future__ import annotations
from datetime import datetime, timezone
from fastapi import FastAPI, Query, UploadFile, File, HTTPException, Header, Depends
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from .db import query
from .object_store import put_bytes, get_bytes
from .auth_core import write_key_valid
import os

app = FastAPI(title="RailGuard-MM API", version="1.0")
_cors = [x.strip() for x in os.getenv("CORS_ORIGINS", "http://localhost:8080").split(",") if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_cors, allow_methods=["GET","POST"], allow_headers=["Content-Type","X-API-Key"])


def require_write_key(x_api_key: str | None = Header(default=None)):
    if not write_key_valid(x_api_key, os.getenv("RAILGUARD_WRITE_API_KEY")):
        raise HTTPException(status_code=401, detail="invalid write API key")

@app.get("/health")
def health(): return {"status":"ok"}

@app.get("/v1/devices")
def devices(): return query("SELECT device_id, max(ts) AS last_seen FROM telemetry GROUP BY device_id ORDER BY device_id")

@app.get("/v1/latest/{device_id}")
def latest(device_id: str):
    rows=query("SELECT * FROM telemetry WHERE device_id=%s ORDER BY ts DESC, seq DESC LIMIT 1",(device_id,)); return rows[0] if rows else {}

@app.get("/v1/series/{device_id}")
def series(device_id: str, minutes: int = Query(30, ge=1, le=1440)):
    measured=query("""SELECT ts,sample_period_ms,vibration_rms,vibration_peak,vision_motion,vision_sharpness,speed_mps,temperature_c,humidity,packet_loss,spool_depth,spool_dropped,camera_matched,sync_error_ms,sensor_skew_ms,clock_alignment_locked,clock_jitter_ms,clock_samples,context_flags,sensor0_rms,sensor1_rms,sensor2_rms FROM telemetry WHERE device_id=%s AND ts >= now()-(%s||' minutes')::interval ORDER BY ts""",(device_id,minutes))
    predicted=query("""
      SELECT p.issued_at,p.source_seq,p.target_ts,p.horizon_steps,p.step_ms,p.vibration_rms_pred,p.vision_motion_pred,
             p.anomaly_probability,p.model_version,
             m.measured_ts,m.vibration_rms_measured,m.vision_motion_measured,m.match_error_ms,
             CASE WHEN m.vibration_rms_measured IS NULL THEN NULL ELSE m.vibration_rms_measured-p.vibration_rms_pred END AS vibration_residual
      FROM predictions p
      LEFT JOIN LATERAL (
        SELECT t.ts AS measured_ts,t.vibration_rms AS vibration_rms_measured,t.vision_motion AS vision_motion_measured,
               abs(extract(epoch from (t.ts-p.target_ts))*1000.0) AS match_error_ms
        FROM telemetry t
        WHERE t.device_id=p.device_id
          AND abs(extract(epoch from (t.ts-p.target_ts))*1000.0) <= greatest(20.0,p.step_ms*0.60)
        ORDER BY abs(extract(epoch from (t.ts-p.target_ts)))
        LIMIT 1
      ) m ON TRUE
      WHERE p.device_id=%s AND p.target_ts >= now()-(%s||' minutes')::interval
      ORDER BY p.target_ts
    """,(device_id,minutes))
    return {"device_id":device_id,"measured":measured,"predicted":predicted}

@app.post("/v1/events/{device_id}", dependencies=[Depends(require_write_key)])
async def upload_event(device_id: str, artifact: UploadFile = File(...)):
    data=await artifact.read()
    if len(data)>25*1024*1024: raise HTTPException(413,"artifact too large")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    safe=(artifact.filename or "artifact.bin").replace("/","_").replace("\\","_")
    name=f"{device_id}/{stamp}_{safe}"
    ref=put_bytes(name,data,artifact.content_type or "application/octet-stream")
    return {"ref":ref,"object_name":name}

@app.get("/v1/events/object/{object_name:path}")
def event_object(object_name: str):
    try: data,ctype=get_bytes(object_name)
    except Exception as e: raise HTTPException(404,str(e))
    return Response(data,media_type=ctype)
