import subprocess
import sys
from pathlib import Path


def test_showcase_experiment_dry_run_writes_plan(tmp_path: Path):
    table = tmp_path / "placeholder.csv"
    out = tmp_path / "showcase"
    proc = subprocess.run(
        [
            sys.executable,
            "ml/run_showcase_experiments.py",
            str(table),
            "--seeds",
            "7,17",
            "--epochs",
            "1",
            "--out",
            str(out),
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert (out / "experiment_manifest.json").exists()
    assert "sensor_only" in proc.stdout
    assert "evaluate_independent_models.py" in proc.stdout
    assert "evaluate_multimodal.py" in proc.stdout
