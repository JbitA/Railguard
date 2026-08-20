from __future__ import annotations
from collections import deque
from pathlib import Path
import numpy as np
import cv2

SENSOR_KEYS=("vibration_rms","vibration_peak","vibration_kurtosis","crest_factor","vision_motion","vision_contrast","speed_mps","temperature_c","humidity")

class DeepTemporalInference:
    """Optional Jetson TorchScript runtime for the multimodal temporal model."""
    def __init__(self, model_path: str | Path, seq_len: int=32, model_version: str='fusion-transformer-v1'):
        import torch
        self.torch=torch; self.model=torch.jit.load(str(model_path),map_location='cuda' if torch.cuda.is_available() else 'cpu').eval(); self.device=next(self.model.parameters(),torch.empty(0)).device if hasattr(self.model,'parameters') else torch.device('cpu')
        self.seq_len=seq_len; self.model_version=model_version; self.frames=deque(maxlen=seq_len); self.sensors=deque(maxlen=seq_len)

    def update(self, frame_bgr: np.ndarray, record: dict) -> dict | None:
        if frame_bgr is None: return None
        rgb=cv2.cvtColor(cv2.resize(frame_bgr,(96,96)),cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        self.frames.append(rgb.transpose(2,0,1))
        v=record['vibration']; vis=record['vision']; g=record['gps']; e=record['environment']
        self.sensors.append(np.array([v['rms_ms2'],v['peak_ms2'],v['kurtosis'],v['crest_factor'],vis['motion_score'],vis['contrast'],g['speed_mps'],e['temperature_c'],e['humidity']],dtype=np.float32))
        if len(self.frames)<self.seq_len: return None
        t=self.torch
        frames=t.from_numpy(np.stack(self.frames)[None]).to(self.device); sensors=t.from_numpy(np.stack(self.sensors)[None]).to(self.device)
        with t.no_grad(): vib,vision,anom=self.model(frames,sensors)
        return {'model_version':self.model_version,'horizons':[1,5,10],'vibration_rms':[float(x) for x in vib[0].cpu().tolist()],'vision_motion':[float(x) for x in vision[0].cpu().tolist()],'anomaly_probability':float(anom[0].cpu())}
