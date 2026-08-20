from __future__ import annotations

import json
from pathlib import Path
from jsonschema import Draft202012Validator, FormatChecker


def load_validator(schema_path: str | Path) -> Draft202012Validator:
    schema = json.loads(Path(schema_path).read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema, format_checker=FormatChecker())


def validate_record(validator: Draft202012Validator, record: dict) -> None:
    errors = sorted(validator.iter_errors(record), key=lambda e: list(e.absolute_path))
    if errors:
        e = errors[0]
        path = '.'.join(map(str, e.absolute_path)) or '<root>'
        raise ValueError(f"telemetry schema violation at {path}: {e.message}")
