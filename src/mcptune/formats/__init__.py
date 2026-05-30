"""Convert DatasetRow collections into the message shapes that fine-tuning
frameworks consume.

Each format is a free function: list[DatasetRow] -> list[dict]. The FORMATS
registry maps a string identifier to its emitter for dynamic dispatch.
"""

from collections.abc import Callable

from mcptune.schema.dataset import DatasetRow

from .anthropic import anthropic_tool_use
from .openai import openai_messages
from .sharegpt import sharegpt
from .trl import trl_sft

FORMATS: dict[str, Callable[[list[DatasetRow]], list[dict]]] = {
    "openai": openai_messages,
    "sharegpt": sharegpt,
    "trl": trl_sft,
    "anthropic": anthropic_tool_use,
}


def convert(rows: list[DatasetRow], format: str) -> list[dict]:
    """Dispatch `rows` to the named format. Raises ValueError for unknown."""
    if format not in FORMATS:
        raise ValueError(f"Unknown format: {format!r}. Available: {sorted(FORMATS)}")
    return FORMATS[format](rows)


__all__ = [
    "convert",
    "openai_messages",
    "sharegpt",
    "trl_sft",
    "anthropic_tool_use",
    "FORMATS",
]
