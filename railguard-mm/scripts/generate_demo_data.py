from __future__ import annotations
import argparse, json, math
from datetime import datetime, timezone, timedelta
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,required=True); p.add_argument("--minutes",type=float,default=5); p.add_argument("--seed",type=int,default=7)
    a=p.parse_args(); rng=np.random.default_rng(a.seed); a.out.parent.mkdir(parents=True,exist_ok=True)
    start=datetime.now(timezone.utc)-timedelta(minutes=a.minutes)
    n=int(a.minutes*60*10)
    with a.out.open("w") as f:
        for i in range(n):
            t=i/10; speed=5.5+1.1*math.sin(t/24)+0.15*rng.normal(); base=1.3+0.13*speed
            event=sum(2.3*math.exp(-0.5*((t-c)/0.8)**2) for c in (55,143,221) if c<a.minutes*60)
            rms=max(0.05,base+0.15*math.sin(t*1.7)+event+0.08*rng.normal())
            motion=max(0.0,0.08*speed+0.1*event+0.03*rng.normal())
            ts=(start+timedelta(seconds=t)).isoformat().replace("+00:00","Z")
            spatial=[max(.01,rms*(1.0+d)+.02*rng.normal()) for d in (-.06,0,.07)]
            record={
              "schema_version":1,"device_id":"railguard-001","ts":ts,"seq":i,"sample_period_ms":100.0,
              "gps":{"lat":40.2395-0.000012*i,"lon":-77.8973+0.000035*i,"speed_mps":speed},
              "environment":{"temperature_c":23.0+0.5*math.sin(t/120),"humidity":0.58+0.02*math.cos(t/90)},
              "vibration":{"rms_ms2":rms,"peak_ms2":rms*(2.8+0.2*rng.random()),"kurtosis":3.0+0.5*event+0.1*rng.normal(),"crest_factor":2.8+0.25*event,"band_energy":[0.12,0.25+0.03*event,0.39+0.04*event,0.24],
                           "sensors":[{"sensor_id":j,"rms_ms2":v,"peak_ms2":v*2.9,"kurtosis":3.0+0.2*event,"crest_factor":2.9,"band_energy":[.12,.25,.39,.24]} for j,v in enumerate(spatial)]},
              "vision":{"motion_score":motion,"contrast":0.68+0.03*rng.normal(),"sharpness":1.4+0.25*event+0.05*rng.normal(),"frame_ref":None},
              "health":{"packet_loss":0,"spool_depth":0,"camera_matched":True,"sync_error_ms":float(rng.normal(0,3)),"sensor_skew_ms":float(abs(rng.normal(4,1))),"clock_alignment_locked":True,"clock_jitter_ms":2.5,"clock_samples":64,"context_flags":3}
            }
            f.write(json.dumps(record)+"\n")
    print(f"wrote {n} records to {a.out}")
if __name__=="__main__": main()
