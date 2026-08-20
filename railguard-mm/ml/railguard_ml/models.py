from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F

class FrameEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2, padding=2), nn.BatchNorm2d(24), nn.SiLU(),
            nn.Conv2d(24, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.SiLU(),
            nn.Conv2d(48, 96, 3, stride=2, padding=1), nn.BatchNorm2d(96), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(96, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class FusionTransformer(nn.Module):
    """Input frames: [B,T,3,H,W], sensors: [B,T,F]."""
    def __init__(self, sensor_dim: int, d_model: int = 128, nhead: int = 4, layers: int = 3, max_steps: int = 256):
        super().__init__()
        self.frame_encoder = FrameEncoder(d_model)
        self.sensor_encoder = nn.Sequential(nn.Linear(sensor_dim, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.fuse = nn.Sequential(nn.Linear(2*d_model, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.pos = nn.Parameter(torch.zeros(1, max_steps, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, dropout=0.1, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(enc, num_layers=layers, enable_nested_tensor=False)
        self.vibration_head = nn.Linear(d_model, 3)  # +1,+5,+10 horizons
        self.vision_head = nn.Linear(d_model, 3)
        self.anomaly_head = nn.Linear(d_model, 1)

    def _forward_encoded(self, vf, sf):
        t = vf.shape[1]
        z = self.fuse(torch.cat([vf, sf], dim=-1)) + self.pos[:, :t]
        z = self.temporal(z)
        hlast = z[:, -1]
        return {
            # These targets are physical non-negative RMS and a [0,1] motion score.
            # Constrain them in the model rather than relying on downstream clipping,
            # which would make training and deployment semantics disagree.
            "vibration": F.softplus(self.vibration_head(hlast)),
            "vision": torch.sigmoid(self.vision_head(hlast)),
            "anomaly_logit": self.anomaly_head(hlast).squeeze(-1),
            "embedding": hlast,
        }

    def forward(self, frames, sensors):
        b,t,c,h,w = frames.shape
        vf = self.frame_encoder(frames.reshape(b*t,c,h,w)).reshape(b,t,-1)
        sf = self.sensor_encoder(sensors)
        return self._forward_encoded(vf, sf)

    def forward_ablated(self, frames, sensors, *, use_vision: bool, use_sensors: bool):
        """Dependency ablation at the learned embedding boundary.

        Masking encoded modalities avoids treating a gray image or a mean-valued raw
        sensor vector as a magically neutral input to biased/normalized encoders.
        This is an analysis method for a trained fusion model, not a separately
        trained unimodal baseline.
        """
        b,t,c,h,w = frames.shape
        vf = self.frame_encoder(frames.reshape(b*t,c,h,w)).reshape(b,t,-1)
        sf = self.sensor_encoder(sensors)
        if not use_vision:
            vf = torch.zeros_like(vf)
        if not use_sensors:
            sf = torch.zeros_like(sf)
        return self._forward_encoded(vf, sf)
