"""Tool-call parsing - the parse side of the format contract.

Whatever the emitters (Issue 2) render as a call, this must read back.
The round-trip test (emit -> parse -> original) is what guarantees the
contract holds. Qwen is first; register more in _PARSERS.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


_QWEN_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def _parse_qwen(text: str) -> list[ToolCall]:
    calls: list[ToolCall] = []
    for match in _QWEN_TOOL_CALL_RE.finditer(text):
        try:
            obj = json.loads(match.group(1).strip())
        except json.JSONDecodeError:
            continue  # malformed body -> skip; loop treats zero calls as final answer
        name = obj.get("name")
        if not isinstance(name, str):
            continue
        args: dict[str, Any] = obj.get("arguments", {})
        calls.append(ToolCall(name=name, arguments=args if isinstance(args, dict) else {}))
    return calls


_PARSERS = {"qwen": _parse_qwen}


def parse_tool_calls(text: str, fmt: str = "qwen") -> list[ToolCall]:
    """Extract tool calls from model output. Empty list == final answer."""
    try:
        parser = _PARSERS[fmt]
    except KeyError:
        raise ValueError(
            f"Unknown tool-call format {fmt!r}. Available: {sorted(_PARSERS)}"
        ) from None
    return parser(text)
