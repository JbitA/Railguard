from __future__ import annotations

import torch

VISUAL_STRUCTURED_COLUMNS = frozenset({"vision_motion", "vision_contrast"})
TRAINING_MODALITIES = ("multimodal", "sensor_only", "vision_only")


def mask_structured_for_modality(
    sensors: torch.Tensor,
    columns: list[str],
    modality: str,
) -> torch.Tensor:
    """Mask normalized structured inputs at their training-mean z-score.

    The data loader has already standardized each feature, so zero is the training
    mean and is the only defensible neutral value for an intentionally absent
    modality.  This function operates before the learned sensor encoder so an
    independently trained baseline never receives the disallowed raw feature.
    """
    if modality not in TRAINING_MODALITIES:
        raise ValueError(f"unknown training modality: {modality}")
    if modality == "multimodal":
        return sensors
    out = sensors.clone()
    for idx, name in enumerate(columns):
        is_visual = name in VISUAL_STRUCTURED_COLUMNS
        keep = (modality == "vision_only" and is_visual) or (modality == "sensor_only" and not is_visual)
        if not keep:
            out[..., idx] = 0.0
    return out


def forward_for_modality(model, frames: torch.Tensor, sensors: torch.Tensor, columns: list[str], modality: str):
    """Forward pass used while *training* a modality-specific checkpoint.

    `sensor_only` removes the camera encoder and engineered visual scalars.
    `vision_only` retains camera frames and visual scalars but removes vibration and
    operating-context features.  `multimodal` receives the complete contract.
    """
    if modality == "multimodal":
        return model(frames, sensors)
    masked = mask_structured_for_modality(sensors, columns, modality)
    if modality == "sensor_only":
        return model.forward_ablated(frames, masked, use_vision=False, use_sensors=True)
    if modality == "vision_only":
        return model.forward_ablated(frames, masked, use_vision=True, use_sensors=True)
    raise ValueError(f"unknown training modality: {modality}")
