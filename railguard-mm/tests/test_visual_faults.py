from pathlib import Path

from PIL import Image
import torch

from ml.railguard_ml.visual_faults import (
    SURFACE_FAULT_CLASSES,
    SurfaceFaultDataset,
    dataset_fingerprint,
    discover_surface_faults,
    stratified_hash_split,
)


def _make_dataset(root: Path, count: int = 6) -> None:
    for class_i, cls in enumerate(SURFACE_FAULT_CLASSES):
        folder = root / cls
        folder.mkdir(parents=True)
        for i in range(count):
            image = Image.new("L", (12, 10), color=(class_i * 20 + i) % 255)
            image.save(folder / f"image_{i}.png")


def test_surface_fault_split_is_disjoint_and_stratified(tmp_path: Path):
    _make_dataset(tmp_path)
    records = discover_surface_faults(tmp_path)
    train, val, test = stratified_hash_split(records)
    all_hashes = {r.sha256 for r in records}
    assert len(all_hashes) == len(records)
    assert not ({r.sha256 for r in train} & {r.sha256 for r in val})
    assert not ({r.sha256 for r in train} & {r.sha256 for r in test})
    assert not ({r.sha256 for r in val} & {r.sha256 for r in test})
    for label in range(len(SURFACE_FAULT_CLASSES)):
        assert any(r.label == label for r in train)
        assert any(r.label == label for r in val)
        assert any(r.label == label for r in test)


def test_surface_fault_dataset_matches_monochrome_deployment_contract(tmp_path: Path):
    _make_dataset(tmp_path)
    records = discover_surface_faults(tmp_path)
    ds = SurfaceFaultDataset(records[:2], image_size=16)
    sample = ds[0]
    assert sample["image"].shape == (3, 16, 16)
    assert torch.allclose(sample["image"][0], sample["image"][1])
    assert torch.allclose(sample["image"][1], sample["image"][2])
    assert len(dataset_fingerprint(records)) == 64
