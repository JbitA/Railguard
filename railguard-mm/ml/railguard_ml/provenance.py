from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def file_sha256(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_image_path(table_path: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute():
        return raw
    # Prepared tables are commonly consumed from the repository root, but making
    # table-relative references work as well keeps exported datasets relocatable.
    beside_table = table_path.parent / raw
    return beside_table if beside_table.exists() else raw


def multimodal_dataset_fingerprint(table_path: str | Path, *, image_column: str = "image_path") -> dict:
    """Fingerprint the *semantic* multimodal training input, including image bytes.

    Hashing only the processed CSV is insufficient because the model dereferences
    image paths at training time.  The combined fingerprint replaces each image path
    with the SHA-256 of that image before hashing a canonical CSV representation.
    This binds structured rows and visual bytes while avoiding absolute-path-specific
    fingerprints when the same dataset is relocated to another machine.
    """
    table_path = Path(table_path)
    table_sha = file_sha256(table_path)
    frame = pd.read_csv(table_path)
    if image_column not in frame.columns:
        return {
            "version": 1,
            "kind": "table-only",
            "sha256": table_sha,
            "table_sha256": table_sha,
            "rows": int(len(frame)),
            "unique_images": 0,
        }

    digests: dict[str, str] = {}
    missing: list[str] = []
    for value in sorted({str(v) for v in frame[image_column].dropna().tolist()}):
        path = _resolve_image_path(table_path, value)
        if not path.is_file():
            missing.append(value)
            continue
        digests[value] = file_sha256(path)
    if missing:
        preview = ", ".join(missing[:3])
        suffix = " ..." if len(missing) > 3 else ""
        raise FileNotFoundError(f"{len(missing)} referenced images are missing: {preview}{suffix}")

    canonical = frame.copy()
    canonical[image_column] = canonical[image_column].map(lambda v: digests.get(str(v), "") if not pd.isna(v) else "")
    canonical_csv = canonical.to_csv(index=False, lineterminator="\n", na_rep="NaN", float_format="%.17g")
    digest = hashlib.sha256(canonical_csv.encode("utf-8")).hexdigest()
    image_digest_list = sorted(set(digests.values()))
    image_set_sha = hashlib.sha256("\n".join(image_digest_list).encode("ascii")).hexdigest()
    return {
        "version": 1,
        "kind": "table+images",
        "sha256": digest,
        "table_sha256": table_sha,
        "rows": int(len(frame)),
        "image_references": int(frame[image_column].notna().sum()),
        "unique_images": int(len(digests)),
        "unique_image_contents": int(len(image_digest_list)),
        "image_set_sha256": image_set_sha,
    }


def verify_dataset_fingerprint(table_path: str | Path, expected: dict) -> dict:
    actual = multimodal_dataset_fingerprint(table_path)
    if not expected or expected.get("version") != actual.get("version"):
        raise ValueError("checkpoint does not contain a compatible multimodal dataset fingerprint")
    if expected.get("sha256") != actual.get("sha256"):
        raise ValueError(
            "multimodal dataset fingerprint mismatch: the structured table and/or referenced image bytes "
            "differ from the checkpoint's training/evaluation dataset"
        )
    return actual


def git_commit(root: str | Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def training_provenance(data_path: str | Path, *, seed: int, deterministic: bool) -> dict:
    fingerprint = multimodal_dataset_fingerprint(data_path)
    return {
        "data_path": str(Path(data_path)),
        "data_sha256": file_sha256(data_path),
        "dataset_fingerprint": fingerprint,
        "seed": int(seed),
        "deterministic_algorithms": bool(deterministic),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "git_commit": git_commit(Path(__file__).resolve().parents[2]),
    }
