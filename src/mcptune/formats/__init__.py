"""Training-format emitters.

Each emitter converts list[DatasetRow] into format-specific dicts.
`convert(rows, fmt, tools=...)` dispatches by name.

- "tool_use": native tool-use (tools in context + structured calls).
  The training default; requires tools=.
- "openai" / "sharegpt" / "trl" / "anthropic": raw conversation shapes
  without tool definitions, kept for users who want them.
"""

from __future__ import annotations

from typing import Any

from mcptune.formats.base import Format

from ..schema.dataset import DatasetRow
from ..schema.tools import ToolSpec
from .anthropic import AnthropicFormat
from .openai import OpenAIFormat
from .sharegpt import ShareGPTFormat
from .tool_use import ToolUseFormat
from .trl import TRLFormat

FORMATS: dict[str, Format] = {
    "tool_use": ToolUseFormat(),
    "openai": OpenAIFormat(),
    "sharegpt": ShareGPTFormat(),
    "trl": TRLFormat(),
    "anthropic": AnthropicFormat(),
}

_NEEDS_TOOLS = {"tool_use"}


def convert(
    rows: list[DatasetRow],
    fmt: str,
    *,
    tools: list[ToolSpec] | None = None,
) -> list[dict[str, Any]]:
    try:
        emitter = FORMATS[fmt]
    except KeyError:
        raise ValueError(f"Unknown format {fmt!r}. Available: {sorted(FORMATS)}") from None

    if emitter.needs_tools and tools is None:
        raise ValueError(
            f"Format {fmt!r} needs the available tools in context; pass tools=discover() result."
        )
    return emitter.format_tool(rows, tools)


__all__ = [
    "FORMATS",
    "convert",
    "ToolUseFormat",
    "OpenAIFormat",
    "ShareGPTFormat",
    "TRLFormat",
    "AnthropicFormat",
]
