from typing import Any


class DatasetValidationError(Exception):
    pass


def validate_dataset_row_dict(data: dict[str, Any]) -> None:
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
