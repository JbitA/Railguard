from __future__ import annotations

from typing import Mapping

from .contracts import DEPLOYMENT_IMAGE_MODE, DEPLOYMENT_SENSOR_COLUMNS, MODEL_ARCH_VERSION
from .objectives import SELECTION_OBJECTIVE_NAME, TRAINING_PROTOCOL_VERSION

EXPECTED_MODALITIES = ("sensor_only", "vision_only", "multimodal")


def _fingerprint_sha(ck: Mapping) -> str | None:
    return (((ck.get("provenance") or {}).get("dataset_fingerprint") or {}).get("sha256"))


def comparable_experiment_contract(ck: Mapping) -> dict:
    """Return fields that must be identical for a fair independent-model comparison."""
    return {
        "model_arch_version": ck.get("model_arch_version"),
        "sensor_columns": list(ck.get("sensor_columns") or []),
        "seq_len": int(ck.get("seq_len", 0)),
        "sample_period_s": float(ck.get("sample_period_s", 0.0)),
        "image_mode": ck.get("image_mode"),
        "split_mode": ck.get("split_mode"),
        "split_seed": ck.get("split_seed"),
        "training_seed": ck.get("training_seed"),
        "training_protocol_version": ck.get("training_protocol_version"),
        "selection_objective_name": ck.get("selection_objective_name"),
        "train_groups": sorted(map(str, ck.get("train_groups") or ck.get("train_runs") or [])),
        "validation_groups": sorted(map(str, ck.get("validation_groups") or ck.get("validation_runs") or [])),
        "test_groups": sorted(map(str, ck.get("test_groups") or ck.get("test_runs") or [])),
        "spatial_block_m": ck.get("spatial_block_m"),
        "spatial_purge_margin_m": ck.get("spatial_purge_margin_m"),
        "dataset_fingerprint_sha256": _fingerprint_sha(ck),
    }


def validate_independent_checkpoint_set(checkpoints: Mapping[str, Mapping]) -> dict:
    """Reject comparisons whose data/split/model contracts are not identical."""
    missing = [m for m in EXPECTED_MODALITIES if m not in checkpoints]
    if missing:
        raise ValueError(f"missing independent checkpoints: {missing}")
    reference = None
    visual_pretraining_sha = None
    for modality in EXPECTED_MODALITIES:
        ck = checkpoints[modality]
        if ck.get("training_modality") != modality:
            raise ValueError(
                f"checkpoint labeled {modality!r} declares training_modality={ck.get('training_modality')!r}"
            )
        if ck.get("model_arch_version") != MODEL_ARCH_VERSION:
            raise ValueError(f"{modality} checkpoint has incompatible model_arch_version")
        if list(ck.get("sensor_columns") or []) != DEPLOYMENT_SENSOR_COLUMNS:
            raise ValueError(f"{modality} checkpoint does not use the deployment feature contract")
        if ck.get("image_mode") != DEPLOYMENT_IMAGE_MODE:
            raise ValueError(f"{modality} checkpoint does not use the deployment image contract")
        if ck.get("training_protocol_version") != TRAINING_PROTOCOL_VERSION:
            raise ValueError(f"{modality} checkpoint has incompatible training_protocol_version")
        if ck.get("selection_objective_name") != SELECTION_OBJECTIVE_NAME:
            raise ValueError(f"{modality} checkpoint uses a different checkpoint-selection objective")
        contract = comparable_experiment_contract(ck)
        if modality in {"vision_only", "multimodal"}:
            visual = ck.get("visual_pretraining") or {}
            sha = visual.get("dataset_sha256")
            if visual_pretraining_sha is None:
                visual_pretraining_sha = sha
            elif sha != visual_pretraining_sha:
                raise ValueError("vision_only and multimodal checkpoints use different visual-pretraining data")
        if not contract["dataset_fingerprint_sha256"]:
            raise ValueError(f"{modality} checkpoint has no multimodal dataset fingerprint")
        if not contract["test_groups"]:
            raise ValueError(f"{modality} checkpoint has no untouched test groups")
        if reference is None:
            reference = contract
        elif contract != reference:
            differing = [k for k in contract if contract[k] != reference[k]]
            raise ValueError(
                f"independent checkpoints are not a controlled comparison; differing fields: {differing}"
            )
    assert reference is not None
    reference = dict(reference)
    reference["visual_pretraining_dataset_sha256"] = visual_pretraining_sha
    return reference
