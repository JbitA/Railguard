import torch
from ml.railguard_ml.models import FusionTransformer

class DeploymentWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.model=model
        self.register_buffer('mean',torch.zeros(1,1,9)); self.register_buffer('std',torch.ones(1,1,9))
    def forward(self,frames,sensors):
        out=self.model(frames,(sensors-self.mean)/(self.std+1e-6))
        return out['vibration'],out['vision'],torch.sigmoid(out['anomaly_logit'])

def test_dynamic_deployment_graph_exports():
    model=DeploymentWrapper(FusionTransformer(sensor_dim=9,d_model=32,nhead=4,layers=1)).eval()
    frames=torch.rand(1,8,3,32,32); sensors=torch.rand(1,8,9)
    ep=torch.export.export(model,(frames,sensors),dynamic_shapes={
        'frames':{1:torch.export.Dim('time',min=4,max=16)},
        'sensors':{1:torch.export.Dim('time',min=4,max=16)},
    })
    vib,vis,anom=ep.module()(frames,sensors)
    assert vib.shape==(1,3) and vis.shape==(1,3) and anom.shape==(1,)
