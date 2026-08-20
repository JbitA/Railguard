from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

SURFACE_FAULT_CLASSES = ("Cracks", "Flakings", "Grooves", "Joints", "Shellings", "Spallings", "Squats")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class VisualFaultRecord:
    path: Path
    label: int
    class_name: str
    sha256: str


def sha256_file(path: Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_surface_faults(root: Path) -> list[VisualFaultRecord]:
    root = root.resolve()
    by_lower = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    records: list[VisualFaultRecord] = []
    for label, class_name in enumerate(SURFACE_FAULT_CLASSES):
        folder = by_lower.get(class_name.lower())
        if folder is None:
            raise ValueError(f"missing expected surface-fault class directory: {class_name}")
        images = sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise ValueError(f"no images found for surface-fault class {class_name}")
        records.extend(VisualFaultRecord(p, label, class_name, sha256_file(p)) for p in images)
    return records


def dataset_fingerprint(records: Iterable[VisualFaultRecord]) -> str:
    rows = sorted(f"{r.class_name}\t{r.sha256}" for r in records)
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


def stratified_hash_split(
    records: list[VisualFaultRecord], *, val_fraction: float = 0.15, test_fraction: float = 0.20
) -> tuple[list[VisualFaultRecord], list[VisualFaultRecord], list[VisualFaultRecord]]:
    if not (0.0 < val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("invalid split fractions")
    train: list[VisualFaultRecord] = []
    val: list[VisualFaultRecord] = []
    test: list[VisualFaultRecord] = []
    for label in range(len(SURFACE_FAULT_CLASSES)):
        class_records = sorted((r for r in records if r.label == label), key=lambda r: r.sha256)
        if len(class_records) < 5:
            raise ValueError(f"surface-fault class {SURFACE_FAULT_CLASSES[label]} is too small for train/val/test")
        n = len(class_records)
        n_test = max(1, int(round(n * test_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1
        test.extend(class_records[:n_test])
        val.extend(class_records[n_test:n_test + n_val])
        train.extend(class_records[n_test + n_val:])
    return train, val, test


class SurfaceFaultDataset(Dataset):
    """Auxiliary railway-surface vision benchmark.

    Images are intentionally converted to luminance and replicated to three channels so
    the encoder sees the same information topology as the deployed monochrome camera.
    """

    def __init__(self, records: list[VisualFaultRecord], *, image_size: int = 96, augment: bool = False):
        self.records = list(records)
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = Image.open(record.path).convert("L").resize((self.image_size, self.image_size))
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if self.augment:
            # Deterministic per-index horizontal reflection avoids hidden RNG state in
            # the data contract while still exposing the encoder to orientation change.
            if index % 2:
                arr = np.ascontiguousarray(arr[:, ::-1])
        arr = np.stack([arr, arr, arr], axis=0)
        return {
            "image": torch.from_numpy(arr),
            "label": torch.tensor(record.label, dtype=torch.long),
            "sha256": record.sha256,
            "path": str(record.path),
        }
