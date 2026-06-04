"""Training-format emitters.

Each emitter converts list[DatasetRow] into format-specific dicts.
`convert(rows, fmt, tools=...)` dispatches by name.

- "tool_use": native tool-use (tools in context + structured calls).
  The training default; requires tools=.
- "openai" / "sharegpt" / "trl" / "anthropic": raw conversation shapes
  without tool definitions, kept for users who want them.
"""

from __future__ import annotations

from ..schema.dataset import DatasetRow
from ..schema.tools import ToolSpec
from .anthropic import anthropic_tool_use
from .openai import openai_messages
from .sharegpt import sharegpt
from .tool_use import tool_use
from .trl import trl_sft

FORMATS = {
    "tool_use": tool_use,
    "openai": openai_messages,
    "sharegpt": sharegpt,
    "trl": trl_sft,
    "anthropic": anthropic_tool_use,
}

_NEEDS_TOOLS = {"tool_use"}


def convert(
    rows: list[DatasetRow],
    fmt: str,
    *,
    tools: list[ToolSpec] | None = None,
) -> list[dict]:
    try:
        emitter = FORMATS[fmt]
    except KeyError:
        raise ValueError(f"Unknown format {fmt!r}. Available: {sorted(FORMATS)}") from None

    if fmt in _NEEDS_TOOLS:
        if tools is None:
            raise ValueError(
                f"Format {fmt!r} needs the available tools in context; "
                "pass tools=discover() result."
            )
        return emitter(rows, tools)
    return emitter(rows)


__all__ = [
    "FORMATS",
    "convert",
    "tool_use",
    "openai_messages",
    "sharegpt",
    "trl_sft",
    "anthropic_tool_use",
]
