from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch import nn
from railguard_ml.models import FusionTransformer
from railguard_ml.contracts import DEPLOYMENT_SENSOR_COLUMNS, MODEL_ARCH_VERSION

class ExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.model=model
        self.register_buffer('mean',mean.reshape(1,1,-1))
        self.register_buffer('std',std.reshape(1,1,-1))
    def forward(self, frames, sensors):
        sensors=(sensors-self.mean)/(self.std+1e-6)
        out=self.model(frames,sensors)
        return out['vibration'], out['vision'], torch.sigmoid(out['anomaly_logit'])

def main():
    p=argparse.ArgumentParser(); p.add_argument('checkpoint',type=Path); p.add_argument('--out',type=Path,default=Path('models/fusion_transformer.ts')); p.add_argument('--seq-len',type=int,default=32); a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False); cols=ck['sensor_columns']
    if ck.get('training_modality', 'multimodal') != 'multimodal':
        raise SystemExit('native multimodal export requires a multimodal checkpoint, not a modality baseline')
    if ck.get('model_arch_version') != MODEL_ARCH_VERSION:
        raise SystemExit(f"checkpoint model_arch_version={ck.get('model_arch_version')} does not match deployment version {MODEL_ARCH_VERSION}")
    if list(cols) != DEPLOYMENT_SENSOR_COLUMNS:
        raise SystemExit('checkpoint sensor_columns do not match native deployment contract')
    model=FusionTransformer(sensor_dim=len(cols)); model.load_state_dict(ck['state_dict']); model.eval()
    mean=torch.tensor(ck.get('sensor_mean',[0.0]*len(cols)),dtype=torch.float32); std=torch.tensor(ck.get('sensor_std',[1.0]*len(cols)),dtype=torch.float32)
    wrapper=ExportWrapper(model,mean,std).eval()
    frames=torch.rand(1,a.seq_len,3,96,96); sensors=torch.rand(1,a.seq_len,len(cols))
    traced=torch.jit.trace(wrapper,(frames,sensors),strict=False,check_trace=False)
    a.out.parent.mkdir(parents=True,exist_ok=True); traced.save(str(a.out)); print(f'saved {a.out}')
if __name__=='__main__': main()
