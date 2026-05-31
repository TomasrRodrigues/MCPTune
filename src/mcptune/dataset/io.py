"""I/O helpers for reading and writing dataset JSONL files.

This module provides small, dependency-free helpers used by the
pipeline to persist and load training datasets produced by MCPTune.
The on-disk format is one JSON object per line with a top-level
``schema_version`` field to allow future compatibility handling.
"""

import json
from dataclasses import asdict
from pathlib import Path

from mcptune.dataset.validate import DatasetValidationError, validate_dataset_row_dict
from mcptune.schema.dataset import DatasetRow

SCHEMA_VERSION = 1


def write_jsonl(rows: list[DatasetRow], path: str | Path) -> None:
    """Write a list of `DatasetRow` objects to a JSONL file.

    Each row is serialized via ``dataclasses.asdict`` and annotated with
    a ``schema_version`` to make future migrations possible.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            record = asdict(row)
            record["schema_version"] = SCHEMA_VERSION
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path, strict: bool = True) -> list[DatasetRow]:
    """Read a JSONL dataset file and return validated `DatasetRow` objects.

    Parameters
    ----------
    path:
        Path to the JSONL file.
    strict:
        If True, validation errors will raise and abort reading. If False,
        invalid rows are skipped.
    """
    path = Path(path)
    rows: list[DatasetRow] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)

            try:
                validate_dataset_row_dict(data)
            except DatasetValidationError:
                if strict:
                    raise
                continue

            row = DatasetRow(
                tool_name=data["tool_name"],
                arguments=data["arguments"],
                request=data["request"],
                response=data.get("response"),
                error=data.get("error"),
                user_intent=data.get("user_intent"),
                intent_prompt_version=data.get("intent_prompt_version"),
            )
            rows.append(row)
    return rows
