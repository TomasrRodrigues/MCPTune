"""Minimal MCP runtime loop.

Lists tools in context, generates, parses the call, executes via the
adapter, feeds the result back, repeats until the model stops calling or
max_turns is hit. Not a production runtime (that's 1.0.0) — feed-back-on-
error is the only recovery, no streaming, no concurrency.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..adapters.base import MCPAdapter
from ..schema.function_schema import toolspecs_to_function_schemas
from ..schema.tools import ToolSpec
from .parsing import ToolCall, parse_tool_calls
from .runner import ModelRunner


@dataclass
class RunResult:
    final_text: str
    messages: list[dict]
    tool_calls: list[ToolCall] = field(default_factory=list)
    stopped_reason: str = "final_answer"  # or "max_turns"


def _extract_text(response: dict) -> str:
    blocks = response.get("content") or []
    texts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
    joined = "\n".join(t for t in texts if t)
    if joined:
        return joined
    sc = response.get("structured_content")
    return json.dumps(sc) if sc is not None else ""


async def _execute(adapter: MCPAdapter, call: ToolCall) -> str:
    try:
        response = await adapter.call_tool(call.name, call.arguments)
    except Exception as e:  # tool/transport failure -> feed back, don't crash
        return f"Error: tool {call.name!r} raised {type(e).__name__}: {e}"
    if response.get("is_error"):
        return f"Error: tool {call.name!r} failed: {_extract_text(response)}"
    return _extract_text(response)


async def run(
    runner: ModelRunner,
    adapter: MCPAdapter,
    user_message: str,
    tools: list[ToolSpec],
    *,
    max_turns: int = 6,
    call_format: str = "qwen",
) -> RunResult:
    function_schemas = toolspecs_to_function_schemas(tools)
    messages: list[dict] = [{"role": "user", "content": user_message}]
    calls_made: list[ToolCall] = []
    text = ""

    for _ in range(max_turns):
        text = runner.generate(messages, function_schemas)
        messages.append({"role": "assistant", "content": text})
        calls = parse_tool_calls(text, fmt=call_format)
        if not calls:
            return RunResult(text, messages, calls_made, "final_answer")
        for call in calls:
            calls_made.append(call)
            result_text = await _execute(adapter, call)
            messages.append({"role": "tool", "name": call.name, "content": result_text})

    return RunResult(text, messages, calls_made, "max_turns")
