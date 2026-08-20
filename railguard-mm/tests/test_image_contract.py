from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

from ml.railguard_ml.contracts import DEPLOYMENT_IMAGE_MODE
from ml.railguard_ml.dataset import RailSequenceDataset
from ml.export_onnx import validate_image_contract


def test_deployment_image_mode_replicates_luminance_into_three_channels(tmp_path):
    image = tmp_path / "color.png"
    Image.new("RGB", (4, 4), color=(255, 0, 0)).save(image)
    rows = [{
        "run_id": "r", "sequence_group_id": "g", "image_path": str(image),
        "vibration_rms": 1.0, "vision_motion": 0.1, "feature": 1.0,
    } for _ in range(12)]
    ds = RailSequenceDataset(pd.DataFrame(rows), ["feature"], seq_len=1, image_size=4, image_mode=DEPLOYMENT_IMAGE_MODE)
    sample = ds[0]["frames"][0].numpy()
    assert np.allclose(sample[0], sample[1]) and np.allclose(sample[1], sample[2])


def test_export_rejects_checkpoint_without_monochrome_deployment_contract():
    validate_image_contract({"image_mode": DEPLOYMENT_IMAGE_MODE})
    try:
        validate_image_contract({"image_mode": "rgb"})
    except ValueError as exc:
        assert "image contract" in str(exc)
    else:
        raise AssertionError("RGB-only checkpoint should be rejected for monochrome deployment")
