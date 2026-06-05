"""Convert internal ToolSpec into the function-schema dicts that model
chat templates expect under `tools=`.

Shared by the runtime (passes tools into the template at inference) and,
later, the format emitters (Issue 2, which render the same definitions
into training context). One converter guarantees the tools the model
trains on match the tools it sees at inference.
"""

from __future__ import annotations

from typing import Any

from .tools import ToolSpec


def toolspec_to_function_schema(tool: ToolSpec) -> dict[str, Any]:
    """Render a ToolSpec as an OpenAI-style function schema.

    Uses raw_input_schema to preserve full JSONSchema fidelity; falls
    back to an empty object schema when the server provided none.
    """
    parameters = tool.raw_input_schema or {"type": "object", "properties": {}}
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


def toolspecs_to_function_schemas(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [toolspec_to_function_schema(t) for t in tools]
