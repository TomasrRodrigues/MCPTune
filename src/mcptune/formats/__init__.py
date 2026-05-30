"""Convert DatasetRow collections into the message shapes that fine-tuning
frameworks consume.

Each format is a free function: list[DatasetRow] -> list[dict]. The FORMATS
registry maps a string identifier to its emitter for dynamic dispatch.
"""

from collections.abc import Callable

from mcptune.schema.dataset import DatasetRow

from .openai import openai_messages

FORMATS: dict[str, Callable[[list[DatasetRow]], list[dict]]] = {
    "openai": openai_messages,
}


def convert(rows: list[DatasetRow], format: str) -> list[dict]:
    """Dispatch `rows` to the named format. Raises ValueError for unknown."""
    if format not in FORMATS:
        raise ValueError(f"Unknown format: {format!r}. Available: {sorted(FORMATS)}")
    return FORMATS[format](rows)


__all__ = ["convert", "openai_messages", "FORMATS"]
