import pytest

torch = pytest.importorskip("torch")
from ml.railguard_ml.models import FusionTransformer

def test_model_shapes():
    m=FusionTransformer(sensor_dim=9,d_model=32,nhead=4,layers=1,max_steps=16)
    out=m(torch.rand(2,8,3,48,48),torch.rand(2,8,9))
    assert out["vibration"].shape==(2,3)
    assert out["vision"].shape==(2,3)
    assert out["anomaly_logit"].shape==(2,)
    assert bool((out["vibration"] >= 0).all())
    assert bool(((out["vision"] >= 0) & (out["vision"] <= 1)).all())


def test_embedding_boundary_ablation_keeps_output_contract():
    m=FusionTransformer(sensor_dim=9,d_model=32,nhead=4,layers=1,max_steps=16)
    frames=torch.rand(2,8,3,48,48); sensors=torch.rand(2,8,9)
    for use_vision,use_sensors in [(False,True),(True,False)]:
        out=m.forward_ablated(frames,sensors,use_vision=use_vision,use_sensors=use_sensors)
        assert out["vibration"].shape==(2,3)
        assert out["vision"].shape==(2,3)
