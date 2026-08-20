from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader

try:
    from railguard_ml.models import FrameEncoder
    from railguard_ml.visual_faults import (
        SURFACE_FAULT_CLASSES,
        SurfaceFaultDataset,
        dataset_fingerprint,
        discover_surface_faults,
        stratified_hash_split,
    )
except ModuleNotFoundError:
    from ml.railguard_ml.models import FrameEncoder
    from ml.railguard_ml.visual_faults import (
        SURFACE_FAULT_CLASSES,
        SurfaceFaultDataset,
        dataset_fingerprint,
        discover_surface_faults,
        stratified_hash_split,
    )

SOURCE_DOI = "10.17632/8hxtgyyxrw.2"
SOURCE_LICENSE = "CC BY 4.0"
IMAGE_CONTRACT = "monochrome_replicated_rgb"


class FaultClassifier(nn.Module):
    def __init__(self, embedding_dim: int = 128):
        super().__init__()
        self.encoder = FrameEncoder(embedding_dim)
        self.head = nn.Linear(embedding_dim, len(SURFACE_FAULT_CLASSES))

    def forward(self, image):
        return self.head(self.encoder(image))


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    truth: list[int] = []
    pred: list[int] = []
    loss_total = 0.0
    count = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            logits = model(image)
            loss_total += float(criterion(logits, label).item())
            count += 1
            truth.extend(label.cpu().tolist())
            pred.extend(logits.argmax(dim=1).cpu().tolist())
    return {
        "loss": loss_total / max(1, count),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(truth, pred, labels=list(range(len(SURFACE_FAULT_CLASSES)))).tolist(),
    }


def markdown(report: dict) -> str:
    lines = [
        "# Auxiliary rail-surface vision benchmark",
        "",
        "> This is an auxiliary representation benchmark, not the headline Rail-VIVID spatial-generalization result. The source consists of frames extracted from inspection video, so near-frame correlation can make a simple image split optimistic.",
        "",
        f"Source: **Railway Track Surface Faults Dataset**, DOI `{report['source']['doi']}`, {report['source']['license']}.",
        "",
        f"Images: **{report['dataset']['images']}**; fingerprint: `{report['dataset']['sha256']}`.",
        "",
        "| Split | Images |",
        "|---|---:|",
        f"| Train | {report['split']['train']} |",
        f"| Validation | {report['split']['validation']} |",
        f"| Test | {report['split']['test']} |",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Test accuracy | {report['test']['accuracy']:.4f} |",
        f"| Test macro F1 | {report['test']['macro_f1']:.4f} |",
        "",
        "The portfolio-relevant experiment is whether initializing the Rail-VIVID vision encoder from this checkpoint improves the untouched-spatial-test multimodal result; classification accuracy here is secondary.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the compact RailGuard frame encoder on the CC-BY railway surface-fault dataset.")
    ap.add_argument("root", type=Path, help="directory containing Cracks/Flakings/Grooves/Joints/Shellings/Spallings/Squats")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out", type=Path, default=Path("models/rail_surface_frame_encoder.pt"))
    ap.add_argument("--metrics", type=Path, default=Path("artifacts/evaluation/visual_faults.json"))
    ap.add_argument("--report", type=Path, default=Path("artifacts/evaluation/visual_faults.md"))
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    records = discover_surface_faults(args.root)
    train_records, val_records, test_records = stratified_hash_split(records)
    train_ds = SurfaceFaultDataset(train_records, augment=True)
    val_ds = SurfaceFaultDataset(val_records)
    test_ds = SurfaceFaultDataset(test_records)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, generator=generator)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaultClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()
    best = float("inf")
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        steps = 0
        for batch in train_loader:
            image = batch["image"].to(device)
            label = batch["label"].to(device)
            loss = criterion(model(image), label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            steps += 1
        val = evaluate(model, val_loader, device)
        print(f"epoch={epoch+1} train_loss={total/max(1,steps):.5f} val_loss={val['loss']:.5f} val_macro_f1={val['macro_f1']:.4f}")
        if val["loss"] < best:
            best = val["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.to(device)
    test = evaluate(model, test_loader, device)
    fingerprint = dataset_fingerprint(records)
    checkpoint = {
        "task": "railway_surface_fault_classification_auxiliary",
        "source_doi": SOURCE_DOI,
        "source_license": SOURCE_LICENSE,
        "dataset_sha256": fingerprint,
        "image_mode": IMAGE_CONTRACT,
        "classes": list(SURFACE_FAULT_CLASSES),
        "training_seed": args.seed,
        "frame_encoder_state_dict": {k.removeprefix("encoder."): v for k, v in best_state.items() if k.startswith("encoder.")},
        "classifier_state_dict": best_state,
        "test_metrics": test,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.out)
    report = {
        "source": {"doi": SOURCE_DOI, "license": SOURCE_LICENSE},
        "dataset": {"images": len(records), "sha256": fingerprint, "classes": list(SURFACE_FAULT_CLASSES)},
        "split": {"train": len(train_records), "validation": len(val_records), "test": len(test_records), "method": "stratified deterministic content-hash split"},
        "test": test,
        "checkpoint": str(args.out),
        "limitation": "video-extracted frames can be temporally correlated; use this as an auxiliary encoder benchmark, not as the headline generalization claim",
    }
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.write_text(json.dumps(report, indent=2) + "\n")
    args.report.write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()
