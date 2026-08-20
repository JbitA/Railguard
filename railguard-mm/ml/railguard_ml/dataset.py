from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

@dataclass
class RunData:
    csv_path: Path
    image_paths: list[Path]


def discover_runs(root: str | Path) -> list[RunData]:
    root = Path(root)
    out = []
    for csv_path in root.rglob("*.csv"):
        imgs = sorted(list(csv_path.parent.rglob("*.jpg")) + list(csv_path.parent.rglob("*.JPG")))
        if imgs:
            out.append(RunData(csv_path, imgs))
    return out


def filename_timestamp(path: Path) -> float | None:
    nums = re.findall(r"\d+(?:\.\d+)?", path.stem)
    if not nums:
        return None
    tok = max(nums, key=len)
    try:
        return float(tok)
    except ValueError:
        return None


class RailSequenceDataset(Dataset):
    """Sequence dataset that never allows a training window to cross a run boundary.

    Sensor normalization is applied only to model inputs; forecast targets stay in
    physical units so exported predictions are directly comparable with measurements.
    """
    def __init__(self, table: pd.DataFrame, sensor_columns: list[str], seq_len: int = 32, image_size: int = 96,
                 sensor_mean: np.ndarray | list[float] | None = None, sensor_std: np.ndarray | list[float] | None = None,
                 image_mode: str = "rgb"):
        self.table = table.reset_index(drop=True)
        self.sensor_columns = sensor_columns
        self.seq_len = seq_len
        self.image_size = image_size
        self.sensor_mean = np.asarray(sensor_mean if sensor_mean is not None else np.zeros(len(sensor_columns)), dtype=np.float32)
        self.sensor_std = np.asarray(sensor_std if sensor_std is not None else np.ones(len(sensor_columns)), dtype=np.float32)
        if image_mode not in {"rgb", "monochrome_replicated_rgb"}:
            raise ValueError(f"unsupported image_mode: {image_mode}")
        self.image_mode = image_mode
        self.sensor_std = np.where(np.abs(self.sensor_std) < 1e-12, 1.0, self.sensor_std)
        self.starts: list[int] = []
        group_column = "sequence_group_id" if "sequence_group_id" in self.table.columns else "run_id"
        if group_column in self.table.columns:
            for _, idxs in self.table.groupby(group_column, sort=False).groups.items():
                positions = sorted(map(int, idxs))
                if not positions:
                    continue
                # Never assume a group remained contiguous after filtering. Split
                # any accidental index gap into a separate span before creating
                # a temporal forecasting window.
                spans: list[list[int]] = []
                current = [positions[0]]
                for pos in positions[1:]:
                    if pos != current[-1] + 1:
                        spans.append(current)
                        current = [pos]
                    else:
                        current.append(pos)
                spans.append(current)
                for span in spans:
                    start, end = span[0], span[-1] + 1
                    stop = end - self.seq_len - 10 + 1
                    for i in range(start, max(start, stop)):
                        self.starts.append(i)
        else:
            self.starts=list(range(max(0,len(self.table)-self.seq_len-10)))

    def __len__(self):
        return len(self.starts)

    def _image(self, path: str) -> torch.Tensor:
        img = Image.open(path)
        if self.image_mode == "monochrome_replicated_rgb":
            img = img.convert("L").convert("RGB")
        else:
            img = img.convert("RGB")
        img = img.resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32).transpose(2,0,1) / 255.0
        return torch.from_numpy(arr)

    def __getitem__(self, idx):
        start=self.starts[idx]
        block = self.table.iloc[start:start+self.seq_len]
        frames = torch.stack([self._image(p) for p in block["image_path"]])
        raw = block[self.sensor_columns].to_numpy(dtype=np.float32)
        sensors = torch.from_numpy((raw-self.sensor_mean)/self.sensor_std)
        now = start + self.seq_len - 1
        target_rows = [self.table.iloc[now+h] for h in (1,5,10)]
        vib = torch.tensor([float(r["vibration_rms"]) for r in target_rows], dtype=torch.float32)
        vis = torch.tensor([float(r["vision_motion"]) for r in target_rows], dtype=torch.float32)
        anomaly_value = self.table.iloc[now].get("anomaly", np.nan)
        anomaly_valid = not pd.isna(anomaly_value)
        anomaly = torch.tensor(float(anomaly_value) if anomaly_valid else 0.0, dtype=torch.float32)
        current_vibration = torch.tensor(float(self.table.iloc[now]["vibration_rms"]), dtype=torch.float32)
        current_vision = torch.tensor(float(self.table.iloc[now]["vision_motion"]), dtype=torch.float32)
        return {"frames": frames, "sensors": sensors, "vibration_target": vib, "vision_target": vis,
                "anomaly": anomaly, "anomaly_valid": torch.tensor(anomaly_valid, dtype=torch.bool),
                "current_vibration": current_vibration, "current_vision": current_vision}
