from __future__ import annotations
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import torch
from torch import nn
try:
    from railguard_ml.models import FusionTransformer
    from railguard_ml.contracts import DEPLOYMENT_IMAGE_MODE, DEPLOYMENT_SENSOR_COLUMNS, MODEL_ARCH_VERSION
    from railguard_ml.provenance import file_sha256
except ModuleNotFoundError:
    from ml.railguard_ml.models import FusionTransformer
    from ml.railguard_ml.contracts import DEPLOYMENT_IMAGE_MODE, DEPLOYMENT_SENSOR_COLUMNS, MODEL_ARCH_VERSION
    from ml.railguard_ml.provenance import file_sha256

def validate_deployment_columns(columns: list[str]) -> None:
    if list(columns) != DEPLOYMENT_SENSOR_COLUMNS:
        raise ValueError(
            "checkpoint sensor_columns do not match native deployment contract; "
            f"expected {DEPLOYMENT_SENSOR_COLUMNS}, got {list(columns)}"
        )


def validate_model_architecture(ckpt: dict) -> None:
    version = ckpt.get("model_arch_version")
    if version != MODEL_ARCH_VERSION:
        raise ValueError(
            f"checkpoint model_arch_version={version!r} does not match deployment version {MODEL_ARCH_VERSION}; retrain with the current model contract"
        )



def validate_image_contract(ckpt: dict) -> None:
    if ckpt.get("image_mode") != DEPLOYMENT_IMAGE_MODE:
        raise ValueError(
            f"checkpoint image_mode={ckpt.get('image_mode')!r} does not match native deployment image contract {DEPLOYMENT_IMAGE_MODE!r}"
        )

def build_deployment_manifest(checkpoint: Path, onnx: Path, ckpt: dict) -> dict:
    checkpoint_sha = file_sha256(checkpoint)
    onnx_sha = file_sha256(onnx)
    model_version = f"fusion-{checkpoint_sha[:12]}"
    return {
        "manifest_version": 1,
        "model_version": model_version,
        "model_arch_version": int(ckpt["model_arch_version"]),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha,
        "onnx": str(onnx),
        "onnx_sha256": onnx_sha,
        "sensor_columns": list(ckpt["sensor_columns"]),
        "image_mode": ckpt["image_mode"],
        "sequence_length": int(ckpt.get("seq_len", 32)),
        "sequence_profile": {"min": 4, "opt": int(ckpt.get("seq_len", 32)), "max": 128},
        "sample_period_ms": float(ckpt.get("sample_period_s", 0.1)) * 1000.0,
        "forecast_horizons_steps": [1, 5, 10],
        "split_mode": ckpt.get("split_mode"),
        "spatial_block_m": ckpt.get("spatial_block_m"),
        "spatial_purge_margin_m": ckpt.get("spatial_purge_margin_m"),
        "spatial_purged_train_rows": ckpt.get("spatial_purged_train_rows", 0),
        "spatial_purged_validation_rows": ckpt.get("spatial_purged_validation_rows", 0),
        "spatial_train_to_heldout_min_m": ckpt.get("spatial_train_to_heldout_min_m"),
        "spatial_validation_to_test_min_m": ckpt.get("spatial_validation_to_test_min_m"),
        "train_groups": ckpt.get("train_groups", []),
        "validation_groups": ckpt.get("validation_groups", []),
        "test_groups": ckpt.get("test_groups", []),
        "training_provenance": ckpt.get("provenance", {}),
        "exported_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


class OnnxWrapper(nn.Module):
    """Deployment wrapper: raw physical sensor features in, physical-unit forecasts out."""
    def __init__(self, model: nn.Module, mean: torch.Tensor, std: torch.Tensor):
        super().__init__(); self.model=model
        self.register_buffer("mean", mean.reshape(1,1,-1)); self.register_buffer("std", std.reshape(1,1,-1))
    def forward(self, frames, sensors):
        out=self.model(frames,(sensors-self.mean)/(self.std+1e-6))
        return out["vibration"],out["vision"],torch.sigmoid(out["anomaly_logit"])

def main():
    p=argparse.ArgumentParser();p.add_argument("checkpoint",type=Path);p.add_argument("--out",type=Path,default=Path("models/fusion_transformer.onnx"));p.add_argument("--opset",type=int,default=18);a=p.parse_args()
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False);validate_model_architecture(ck);validate_image_contract(ck)
    if ck.get("training_modality", "multimodal") != "multimodal":
        raise SystemExit("native multimodal export requires a multimodal checkpoint, not a modality baseline")
    cols=ck["sensor_columns"];validate_deployment_columns(cols);seq_len=int(ck.get("seq_len",32))
    model=FusionTransformer(sensor_dim=len(cols));model.load_state_dict(ck["state_dict"]);model.eval()
    wrapper=OnnxWrapper(model,torch.tensor(ck.get("sensor_mean",[0.0]*len(cols))),torch.tensor(ck.get("sensor_std",[1.0]*len(cols)))).eval()
    frames=torch.rand(1,seq_len,3,96,96);sensors=torch.rand(1,seq_len,len(cols))
    a.out.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(wrapper,(frames,sensors),str(a.out),input_names=["frames","sensors"],output_names=["vibration","vision","anomaly_probability"],opset_version=a.opset,dynamo=True,
                      dynamic_shapes={"frames":{1:torch.export.Dim("time",min=4,max=128)},"sensors":{1:torch.export.Dim("time",min=4,max=128)}})
    manifest = build_deployment_manifest(a.checkpoint, a.out, ck)
    manifest_path = Path(str(a.out) + ".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"saved {a.out} and {manifest_path} model_version={manifest['model_version']} (sequence profile 4..128; train/default={seq_len})")
if __name__=="__main__":main()
