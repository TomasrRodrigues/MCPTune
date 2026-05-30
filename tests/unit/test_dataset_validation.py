import pytest

from mcptune.dataset.io import read_jsonl
from mcptune.dataset.validate import (
    DatasetValidationError,
    validate_dataset_row_dict,
)

# ----------------------------
# VALID CASE
# ----------------------------


def test_valid_dataset_row_passes():
    data = {
        "tool_name": "weather_tool",
        "arguments": {"city": "Porto"},
        "request": {"prompt": "Weather in Porto"},
    }

    validate_dataset_row_dict(data)


# ----------------------------
# MISSING FIELD CASES
# ----------------------------


def test_missing_tool_name_fails():
    data = {
        "arguments": {"city": "Porto"},
        "request": {"prompt": "Weather in Porto"},
    }

    with pytest.raises(DatasetValidationError):
        validate_dataset_row_dict(data)


def test_missing_arguments_fails():
    data = {
        "tool_name": "weather_tool",
        "request": {"prompt": "Weather in Porto"},
    }

    with pytest.raises(DatasetValidationError):
        validate_dataset_row_dict(data)


def test_missing_request_fails():
    data = {
        "tool_name": "weather_tool",
        "arguments": {"city": "Porto"},
    }

    with pytest.raises(DatasetValidationError):
        validate_dataset_row_dict(data)


# ----------------------------
# TYPE SAFETY CASES
# ----------------------------


def test_tool_name_must_be_string():
    data = {
        "tool_name": 123,
        "arguments": {"city": "Porto"},
        "request": {"prompt": "Weather in Porto"},
    }

    with pytest.raises(DatasetValidationError):
        validate_dataset_row_dict(data)


def test_arguments_must_be_dict():
    data = {
        "tool_name": "weather_tool",
        "arguments": "not-a-dict",
        "request": {"prompt": "Weather in Porto"},
    }

    with pytest.raises(DatasetValidationError):
        validate_dataset_row_dict(data)


def test_request_must_be_dict():
    data = {
        "tool_name": "weather_tool",
        "arguments": {"city": "Porto"},
        "request": "not-a-dict",
    }

    with pytest.raises(DatasetValidationError):
        validate_dataset_row_dict(data)


# ----------------------------
# OPTIONAL FIELDS (future-proofing)
# ----------------------------


def test_response_optional_valid_types():
    for response in [
        None,
        {"temp": 20},
        "ok",
        123,
        True,
        [1, 2, 3],
    ]:
        data = {
            "tool_name": "weather_tool",
            "arguments": {"city": "Porto"},
            "request": {"prompt": "Weather in Porto"},
            "response": response,
        }

        validate_dataset_row_dict(data)


def test_invalid_response_type_fails():
    class BadObject:
        pass

    data = {
        "tool_name": "weather_tool",
        "arguments": {"city": "Porto"},
        "request": {"prompt": "Weather in Porto"},
        "response": BadObject(),
    }

    with pytest.raises(DatasetValidationError):
        validate_dataset_row_dict(data)


def test_error_must_be_string_or_none():
    # valid
    validate_dataset_row_dict(
        {
            "tool_name": "x",
            "arguments": {},
            "request": {},
            "error": None,
        }
    )

    validate_dataset_row_dict(
        {
            "tool_name": "x",
            "arguments": {},
            "request": {},
            "error": "something went wrong",
        }
    )


def test_invalid_error_type_fails():
    data = {
        "tool_name": "x",
        "arguments": {},
        "request": {},
        "error": {"msg": "bad"},
    }

    with pytest.raises(DatasetValidationError):
        validate_dataset_row_dict(data)


# ----------------------------
# INTEGRATION WITH IO LAYER
# ----------------------------


def test_read_jsonl_rejects_invalid_data(tmp_path):
    file = tmp_path / "bad.jsonl"

    file.write_text('{"tool_name": "weather_tool", "arguments": "not-a-dict", "request": {}}\n')

    with pytest.raises(DatasetValidationError):
        read_jsonl(file)


@pytest.mark.unit
def test_user_intent_optional_string_passes():
    validate_dataset_row_dict(
        {
            "tool_name": "x",
            "arguments": {},
            "request": {},
            "user_intent": "Show me the weather in Lisbon",
        }
    )


@pytest.mark.unit
def test_user_intent_none_passes():
    validate_dataset_row_dict(
        {
            "tool_name": "x",
            "arguments": {},
            "request": {},
            "user_intent": None,
        }
    )


@pytest.mark.unit
def test_user_intent_non_string_fails():
    with pytest.raises(DatasetValidationError):
        validate_dataset_row_dict(
            {
                "tool_name": "x",
                "arguments": {},
                "request": {},
                "user_intent": 42,
            }
        )


@pytest.mark.unit
def test_intent_prompt_version_optional_string_passes():
    validate_dataset_row_dict(
        {
            "tool_name": "x",
            "arguments": {},
            "request": {},
            "intent_prompt_version": "intent_v1",
        }
    )


@pytest.mark.unit
def test_intent_prompt_version_non_string_fails():
    with pytest.raises(DatasetValidationError):
        validate_dataset_row_dict(
            {
                "tool_name": "x",
                "arguments": {},
                "request": {},
                "intent_prompt_version": {"version": 1},
            }
        )
