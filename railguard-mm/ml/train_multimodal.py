from __future__ import annotations
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
try:
    from railguard_ml.dataset import RailSequenceDataset
    from railguard_ml.models import FusionTransformer
    from railguard_ml.splits import grouped_train_val_test_split, spatial_train_val_test_split
    from railguard_ml.contracts import DEPLOYMENT_IMAGE_MODE, DEPLOYMENT_SENSOR_COLUMNS, MODEL_ARCH_VERSION
    from railguard_ml.provenance import training_provenance
    from railguard_ml.validation import validate_modeling_table
    from railguard_ml.modalities import TRAINING_MODALITIES, forward_for_modality
    from railguard_ml.objectives import ANOMALY_WEIGHT, SELECTION_OBJECTIVE_NAME, TRAINING_PROTOCOL_VERSION, forecast_loss
except ModuleNotFoundError:
    from ml.railguard_ml.dataset import RailSequenceDataset
    from ml.railguard_ml.models import FusionTransformer
    from ml.railguard_ml.splits import grouped_train_val_test_split, spatial_train_val_test_split
    from ml.railguard_ml.contracts import DEPLOYMENT_IMAGE_MODE, DEPLOYMENT_SENSOR_COLUMNS, MODEL_ARCH_VERSION
    from ml.railguard_ml.provenance import training_provenance
    from ml.railguard_ml.validation import validate_modeling_table
    from ml.railguard_ml.modalities import TRAINING_MODALITIES, forward_for_modality
    from ml.railguard_ml.objectives import ANOMALY_WEIGHT, SELECTION_OBJECTIVE_NAME, TRAINING_PROTOCOL_VERSION, forecast_loss

SENSOR_COLUMNS = DEPLOYMENT_SENSOR_COLUMNS



def evaluate(model, dl, device, huber, *, modality: str = "multimodal"):
    model.eval()
    total = 0.0
    vibration_total = 0.0
    vision_total = 0.0
    count = 0
    vibration_mae = torch.zeros(3, device=device)
    vision_mae = torch.zeros(3, device=device)
    with torch.no_grad():
        for batch in dl:
            frames = batch["frames"].to(device)
            sensors = batch["sensors"].to(device)
            vibration_target = batch["vibration_target"].to(device)
            vision_target = batch["vision_target"].to(device)
            out = forward_for_modality(model, frames, sensors, SENSOR_COLUMNS, modality)
            objective, vibration_loss, vision_loss = forecast_loss(
                out, vibration_target, vision_target, huber
            )
            total += float(objective.item())
            vibration_total += float(vibration_loss.item())
            vision_total += float(vision_loss.item())
            count += 1
            vibration_mae += torch.abs(out["vibration"] - vibration_target).sum(dim=0)
            vision_mae += torch.abs(out["vision"] - vision_target).sum(dim=0)
    n = max(1, len(dl.dataset))
    return {
        "selection_objective": total / max(1, count),
        "vibration_huber": vibration_total / max(1, count),
        "vision_huber": vision_total / max(1, count),
        "vibration_horizon_mae": (vibration_mae / n).detach().cpu().tolist(),
        "vision_horizon_mae": (vision_mae / n).detach().cpu().tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("table", type=Path, help="processed CSV with image_path and synchronized features")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--test-fraction", type=float, default=0.20)
    p.add_argument("--split-seed", type=int, default=7)
    p.add_argument("--seed", type=int, default=17, help="model/data-loader random seed")
    p.add_argument("--deterministic", action="store_true", help="request deterministic PyTorch algorithms where supported")
    p.add_argument("--split-mode", choices=["spatial", "run"], default="spatial", help="spatial prevents repeated-route location leakage; run is a secondary robustness split")
    p.add_argument("--spatial-block-m", type=float, default=500.0)
    p.add_argument("--spatial-purge-margin-m", type=float, default=30.0, help="minimum geodesic separation between spatial train/validation/test rows")
    p.add_argument("--sample-period-s", type=float, default=0.1, help="model time-step duration used to interpret +1/+5/+10 horizons")
    p.add_argument("--seq-len", type=int, default=32)
    p.add_argument("--modality", choices=TRAINING_MODALITIES, default="multimodal", help="train a full fusion model or a separately optimized unimodal baseline")
    p.add_argument("--frame-encoder-init", type=Path, default=None, help="optional auxiliary visual-pretraining checkpoint produced by train_visual_faults.py")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    if args.out is None:
        suffix = "" if args.modality == "multimodal" else f"_{args.modality}"
        args.out = Path(f"models/fusion_transformer{suffix}.pt")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
    provenance = training_provenance(args.table, seed=args.seed, deterministic=args.deterministic)
    df = pd.read_csv(args.table)
    df = df.sort_values(['run_id','ts'] if 'run_id' in df.columns else ['ts']).reset_index(drop=True)
    validate_modeling_table(df, SENSOR_COLUMNS)
    if args.split_mode == "spatial":
        split=spatial_train_val_test_split(df,block_size_m=args.spatial_block_m,val_fraction=args.val_fraction,test_fraction=args.test_fraction,seed=args.split_seed,purge_margin_m=args.spatial_purge_margin_m)
        train_df,val_df=split.train,split.validation
        val_groups,test_groups=split.validation_blocks,split.test_blocks
        train_groups=split.train_blocks
    else:
        split=grouped_train_val_test_split(df,val_fraction=args.val_fraction,test_fraction=args.test_fraction,seed=args.split_seed)
        train_df,val_df=split.train,split.validation
        val_groups,test_groups=split.validation_runs,split.test_runs
        train_groups=split.train_runs
    if len(train_df) < args.seq_len+11 or len(val_df) < args.seq_len+11:
        raise SystemExit('Not enough windows after train/validation split; combine more runs or reduce --seq-len.')
    labeled_anomaly = train_df["anomaly"].dropna() if "anomaly" in train_df.columns else pd.Series(dtype=float)
    use_anomaly = labeled_anomaly.nunique() > 1
    sensor_mean = train_df[SENSOR_COLUMNS].mean()
    sensor_std = train_df[SENSOR_COLUMNS].std().replace(0, 1.0)
    train_ds=RailSequenceDataset(train_df,SENSOR_COLUMNS,seq_len=args.seq_len,sensor_mean=sensor_mean.to_numpy(),sensor_std=sensor_std.to_numpy(),image_mode=DEPLOYMENT_IMAGE_MODE)
    val_ds=RailSequenceDataset(val_df,SENSOR_COLUMNS,seq_len=args.seq_len,sensor_mean=sensor_mean.to_numpy(),sensor_std=sensor_std.to_numpy(),image_mode=DEPLOYMENT_IMAGE_MODE)
    loader_generator=torch.Generator().manual_seed(args.seed)
    train_dl=DataLoader(train_ds,batch_size=args.batch,shuffle=True,num_workers=2,generator=loader_generator)
    val_dl=DataLoader(val_ds,batch_size=args.batch,shuffle=False,num_workers=2)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=FusionTransformer(sensor_dim=len(SENSOR_COLUMNS)).to(device)
    visual_pretraining = None
    if args.frame_encoder_init is not None:
        if args.modality == "sensor_only":
            raise SystemExit("--frame-encoder-init is meaningless for sensor_only training")
        init = torch.load(args.frame_encoder_init, map_location="cpu", weights_only=False)
        state = init.get("frame_encoder_state_dict")
        if not isinstance(state, dict) or not state:
            raise SystemExit("visual-pretraining checkpoint does not contain frame_encoder_state_dict")
        model.frame_encoder.load_state_dict(state, strict=True)
        visual_pretraining = {
            "checkpoint": str(args.frame_encoder_init),
            "source_doi": init.get("source_doi"),
            "source_license": init.get("source_license"),
            "dataset_sha256": init.get("dataset_sha256"),
            "image_mode": init.get("image_mode"),
        }
        if visual_pretraining["image_mode"] != DEPLOYMENT_IMAGE_MODE:
            raise SystemExit("visual-pretraining image contract does not match the deployment image contract")
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    huber=nn.SmoothL1Loss(); bce=nn.BCEWithLogitsLoss(reduction="none"); best=float('inf')
    args.out.parent.mkdir(parents=True,exist_ok=True)
    print(f'modality={args.modality} train_windows={len(train_ds)} val_windows={len(val_ds)} split={args.split_mode} validation_groups={len(val_groups)} untouched_test_groups={len(test_groups)}' + (f' purge_margin_m={split.purge_margin_m} train_to_heldout_min_m={split.train_to_heldout_min_m:.2f} validation_to_test_min_m={split.validation_to_test_min_m:.2f} purged_train_rows={split.purged_train_rows} purged_validation_rows={split.purged_validation_rows}' if args.split_mode == 'spatial' else ''))
    for epoch in range(args.epochs):
        model.train(); running=0.0
        for batch in train_dl:
            frames=batch["frames"].to(device); sensors=batch["sensors"].to(device); out=forward_for_modality(model,frames,sensors,SENSOR_COLUMNS,args.modality)
            loss, _, _ = forecast_loss(out, batch["vibration_target"].to(device), batch["vision_target"].to(device), huber)
            if use_anomaly:
                valid=batch["anomaly_valid"].to(device)
                if bool(valid.any()):
                    per_item=bce(out["anomaly_logit"],batch["anomaly"].to(device))
                    loss=loss+ANOMALY_WEIGHT*per_item[valid].mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step(); running += float(loss.item())
        validation=evaluate(model,val_dl,device,huber,modality=args.modality)
        val_loss=validation["selection_objective"]
        print(f"epoch={epoch+1} train_loss={running/max(1,len(train_dl)):.5f} val_forecast_objective={val_loss:.5f} val_vib_huber={validation["vibration_huber"]:.5f} val_vision_huber={validation["vision_huber"]:.5f} vibration_horizon_mae={validation["vibration_horizon_mae"]} vision_horizon_mae={validation["vision_horizon_mae"]}")
        if val_loss < best:
            best=val_loss
            torch.save({"state_dict":model.state_dict(),"model_arch_version":MODEL_ARCH_VERSION,"training_modality":args.modality,"sensor_columns":SENSOR_COLUMNS,"sensor_mean":sensor_mean.tolist(),"sensor_std":sensor_std.tolist(),"seq_len":args.seq_len,"split_mode":args.split_mode,"validation_groups":val_groups,"test_groups":test_groups,"train_groups":train_groups,"spatial_block_m":args.spatial_block_m if args.split_mode=="spatial" else None,"spatial_purge_margin_m":args.spatial_purge_margin_m if args.split_mode=="spatial" else None,"spatial_purged_train_rows":split.purged_train_rows if args.split_mode=="spatial" else 0,"spatial_purged_validation_rows":split.purged_validation_rows if args.split_mode=="spatial" else 0,"spatial_train_to_heldout_min_m":split.train_to_heldout_min_m if args.split_mode=="spatial" else None,"spatial_validation_to_test_min_m":split.validation_to_test_min_m if args.split_mode=="spatial" else None,"validation_runs":val_groups if args.split_mode=="run" else [],"test_runs":test_groups if args.split_mode=="run" else [],"train_runs":train_groups if args.split_mode=="run" else [],"split_seed":args.split_seed,"training_seed":args.seed,"training_protocol_version":TRAINING_PROTOCOL_VERSION,"selection_objective_name":SELECTION_OBJECTIVE_NAME,"image_mode":DEPLOYMENT_IMAGE_MODE,"sample_period_s":args.sample_period_s,"anomaly_supervision":use_anomaly,"visual_pretraining":visual_pretraining,"val_forecast_objective":validation["selection_objective"],"val_vibration_huber":validation["vibration_huber"],"val_vision_huber":validation["vision_huber"],"vibration_horizon_mae":validation["vibration_horizon_mae"],"vision_horizon_mae":validation["vision_horizon_mae"],"provenance":provenance},args.out)
    print(f'saved best checkpoint to {args.out}')

if __name__ == "__main__": main()
