import copy

import pytest
import torch

from ml.railguard_ml.contracts import DEPLOYMENT_IMAGE_MODE, DEPLOYMENT_SENSOR_COLUMNS, MODEL_ARCH_VERSION
from ml.railguard_ml.modalities import mask_structured_for_modality
from ml.railguard_ml.objectives import SELECTION_OBJECTIVE_NAME, TRAINING_PROTOCOL_VERSION
from ml.railguard_ml.model_comparison import validate_independent_checkpoint_set


def test_raw_modality_masks_remove_disallowed_structured_features():
    x = torch.arange(len(DEPLOYMENT_SENSOR_COLUMNS), dtype=torch.float32).reshape(1, 1, -1) + 1
    sensor = mask_structured_for_modality(x, DEPLOYMENT_SENSOR_COLUMNS, "sensor_only")
    vision = mask_structured_for_modality(x, DEPLOYMENT_SENSOR_COLUMNS, "vision_only")
    for i, name in enumerate(DEPLOYMENT_SENSOR_COLUMNS):
        if name in {"vision_motion", "vision_contrast"}:
            assert sensor[..., i].item() == 0.0
            assert vision[..., i].item() == x[..., i].item()
        else:
            assert sensor[..., i].item() == x[..., i].item()
            assert vision[..., i].item() == 0.0


def _checkpoint(modality: str):
    return {
        "training_modality": modality,
        "model_arch_version": MODEL_ARCH_VERSION,
        "sensor_columns": list(DEPLOYMENT_SENSOR_COLUMNS),
        "seq_len": 32,
        "sample_period_s": 0.1,
        "image_mode": DEPLOYMENT_IMAGE_MODE,
        "split_mode": "spatial",
        "split_seed": 7,
        "training_protocol_version": TRAINING_PROTOCOL_VERSION,
        "training_seed": 17,
        "selection_objective_name": SELECTION_OBJECTIVE_NAME,
        "train_groups": ["a"],
        "validation_groups": ["b"],
        "test_groups": ["c"],
        "spatial_block_m": 500.0,
        "spatial_purge_margin_m": 30.0,
        "provenance": {"dataset_fingerprint": {"sha256": "a" * 64}},
    }


def test_independent_checkpoint_comparison_requires_identical_experiment_contract():
    cks = {m: _checkpoint(m) for m in ("sensor_only", "vision_only", "multimodal")}
    contract = validate_independent_checkpoint_set(cks)
    assert contract["test_groups"] == ["c"]
    bad = copy.deepcopy(cks)
    bad["vision_only"]["test_groups"] = ["different"]
    with pytest.raises(ValueError, match="controlled comparison"):
        validate_independent_checkpoint_set(bad)


def test_visual_pretraining_contract_must_match_between_visual_models():
    cks = {m: _checkpoint(m) for m in ("sensor_only", "vision_only", "multimodal")}
    cks["vision_only"]["visual_pretraining"] = {"dataset_sha256": "b" * 64}
    cks["multimodal"]["visual_pretraining"] = {"dataset_sha256": "b" * 64}
    contract = validate_independent_checkpoint_set(cks)
    assert contract["visual_pretraining_dataset_sha256"] == "b" * 64

    bad = copy.deepcopy(cks)
    bad["multimodal"]["visual_pretraining"] = {"dataset_sha256": "c" * 64}
    with pytest.raises(ValueError, match="different visual-pretraining data"):
        validate_independent_checkpoint_set(bad)
