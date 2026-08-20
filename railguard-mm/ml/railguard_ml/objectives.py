from __future__ import annotations

import torch
from torch import nn

VIBRATION_FORECAST_WEIGHT = 1.0
VISION_FORECAST_WEIGHT = 0.4
ANOMALY_WEIGHT = 0.5
SELECTION_OBJECTIVE_NAME = "forecast_huber_vibration_plus_0.4_vision"
TRAINING_PROTOCOL_VERSION = 3


def forecast_loss(
    outputs: dict[str, torch.Tensor],
    vibration_target: torch.Tensor,
    vision_target: torch.Tensor,
    huber: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    vibration = huber(outputs["vibration"], vibration_target)
    vision = huber(outputs["vision"], vision_target)
    total = VIBRATION_FORECAST_WEIGHT * vibration + VISION_FORECAST_WEIGHT * vision
    return total, vibration, vision
