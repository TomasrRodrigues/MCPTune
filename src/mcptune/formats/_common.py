"""Shared helpers for format emitters."""

import json
from typing import Any

from mcptune.schema.dataset import DatasetRow


def template_intent(row: DatasetRow) -> str:
    """Fallback when a DatasetRow has no user_intent. Rows from
    MCPTune.build_dataset always have one; this only fires when callers
    construct DatasetRow manually."""
    if row.arguments:
        return f"Use the {row.tool_name} tool with arguments: {json.dumps(row.arguments)}"
    return f"Use the {row.tool_name} tool."


def result_to_text(response: Any) -> str:
    if not isinstance(response, dict):
        return "" if response is None else str(response)
    blocks = response.get("content") or []
    texts = [
        b["text"]
        for b in blocks
        if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
    ]
    if texts:
        return "\n".join(t for t in texts if t)
    sc = response.get("structured_content")
    return json.dumps(sc) if sc is not None else ""
