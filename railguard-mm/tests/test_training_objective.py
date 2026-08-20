from __future__ import annotations

import torch
from torch import nn

from ml.railguard_ml.objectives import (
    SELECTION_OBJECTIVE_NAME,
    TRAINING_PROTOCOL_VERSION,
    VISION_FORECAST_WEIGHT,
    forecast_loss,
)


def test_forecast_selection_objective_matches_documented_training_weights():
    outputs = {
        "vibration": torch.tensor([[2.0, 2.0, 2.0]]),
        "vision": torch.tensor([[0.2, 0.2, 0.2]]),
    }
    vibration_target = torch.tensor([[1.0, 1.0, 1.0]])
    vision_target = torch.tensor([[0.0, 0.0, 0.0]])
    huber = nn.SmoothL1Loss()
    total, vibration, vision = forecast_loss(outputs, vibration_target, vision_target, huber)
    assert torch.allclose(total, vibration + VISION_FORECAST_WEIGHT * vision)
    assert TRAINING_PROTOCOL_VERSION >= 2
    assert "vision" in SELECTION_OBJECTIVE_NAME
