"""Validation helpers for dataset rows.

Provides a small, explicit validation routine used when loading JSONL
dataset files. The checks are intentionally conservative: they verify
presence and types of required fields and ensure optional fields have
acceptable primitive types.
"""

from typing import Any


class DatasetValidationError(Exception):
    """Raised when a dataset row fails validation."""


def validate_dataset_row_dict(data: dict[str, Any]) -> None:
    """Validate a deserialized dataset row dictionary.

    Raises ``DatasetValidationError`` on any validation failure.
    """
    required_fields = ["tool_name", "arguments", "request"]

    for field in required_fields:
        if field not in data:
            raise DatasetValidationError(f"Missing required field: {field}")

    if not isinstance(data["tool_name"], str):
        raise DatasetValidationError("tool_name must be str")

    if not isinstance(data["arguments"], dict):
        raise DatasetValidationError("arguments must be dict")

    if not isinstance(data["request"], dict):
        raise DatasetValidationError("request must be dict")

    if "response" in data and data["response"] is not None:
        if not isinstance(data["response"], (dict, list, str, int, float, bool)):
            raise DatasetValidationError("response has invalid type")

    if "error" in data and data["error"] is not None:
        if not isinstance(data["error"], str):
            raise DatasetValidationError("error must be str or None")

    if "user_intent" in data and data["user_intent"] is not None:
        if not isinstance(data["user_intent"], str):
            raise DatasetValidationError("user_intent must be str or None")

    if "intent_prompt_version" in data and data["intent_prompt_version"] is not None:
        if not isinstance(data["intent_prompt_version"], str):
            raise DatasetValidationError("intent_prompt_version must be str or None")
