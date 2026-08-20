from pathlib import Path

from ml.railguard_ml.provenance import (
    file_sha256,
    multimodal_dataset_fingerprint,
    training_provenance,
    verify_dataset_fingerprint,
)
import pandas as pd
import pytest


def test_training_provenance_hashes_exact_input_artifact(tmp_path: Path):
    data = tmp_path / "table.csv"
    data.write_text("a,b\n1,2\n")
    first = file_sha256(data)
    manifest = training_provenance(data, seed=23, deterministic=True)
    assert manifest["data_sha256"] == first
    assert manifest["seed"] == 23
    assert manifest["deterministic_algorithms"] is True
    data.write_text("a,b\n1,3\n")
    assert file_sha256(data) != first


def test_multimodal_fingerprint_binds_referenced_image_bytes_and_is_path_independent(tmp_path: Path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir(); b.mkdir()
    image_a = a / "frame.bin"
    image_b = b / "frame.bin"
    image_a.write_bytes(b"same-image")
    image_b.write_bytes(b"same-image")
    table_a = a / "table.csv"
    table_b = b / "table.csv"
    pd.DataFrame({"ts": [1], "image_path": [str(image_a)], "vibration_rms": [2.0]}).to_csv(table_a, index=False)
    pd.DataFrame({"ts": [1], "image_path": [str(image_b)], "vibration_rms": [2.0]}).to_csv(table_b, index=False)

    fp_a = multimodal_dataset_fingerprint(table_a)
    fp_b = multimodal_dataset_fingerprint(table_b)
    assert fp_a["sha256"] == fp_b["sha256"]
    assert fp_a["table_sha256"] != fp_b["table_sha256"]  # raw path strings differ

    image_b.write_bytes(b"changed-image")
    changed = multimodal_dataset_fingerprint(table_b)
    assert changed["sha256"] != fp_a["sha256"]
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        verify_dataset_fingerprint(table_b, fp_a)


def test_multimodal_fingerprint_rejects_missing_images(tmp_path: Path):
    table = tmp_path / "table.csv"
    pd.DataFrame({"image_path": [str(tmp_path / "missing.jpg")], "x": [1]}).to_csv(table, index=False)
    with pytest.raises(FileNotFoundError, match="referenced images are missing"):
        multimodal_dataset_fingerprint(table)
