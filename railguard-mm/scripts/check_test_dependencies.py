"""Fail fast when the clean test environment is missing an implicitly required package.

The project deliberately keeps service-container requirements separate from the core
host test environment.  This smoke check covers packages imported by the host-side ML,
edge and cloud-contract tests so a developer machine with ambient packages cannot hide
an incomplete ``requirements-dev.txt``.
"""
from __future__ import annotations

import importlib

REQUIRED_IMPORTS = {
    "cv2": "opencv-python-headless",
    "jsonschema": "jsonschema",
    "huggingface_hub": "huggingface-hub",
    "matplotlib": "matplotlib",
    "numpy": "numpy",
    "pandas": "pandas",
    "PIL": "Pillow",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "torch": "torch",
    "yaml": "PyYAML",
}


def main() -> int:
    missing: list[str] = []
    for module, package in REQUIRED_IMPORTS.items():
        try:
            importlib.import_module(module)
        except Exception as exc:  # pragma: no cover - only exercised in broken envs
            missing.append(f"{package} ({module}: {exc})")
    if missing:
        raise SystemExit("missing host-test dependencies: " + "; ".join(missing))
    print(f"host-test dependency smoke: PASS ({len(REQUIRED_IMPORTS)} imports)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
