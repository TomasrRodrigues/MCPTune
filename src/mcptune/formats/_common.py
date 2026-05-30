"""Shared helpers for format emitters."""

import json

from mcptune.schema.dataset import DatasetRow


def template_intent(row: DatasetRow) -> str:
    """Fallback when a DatasetRow has no user_intent. Rows from
    MCPTune.build_dataset always have one; this only fires when callers
    construct DatasetRow manually."""
    if row.arguments:
        return f"Use the {row.tool_name} tool with arguments: {json.dumps(row.arguments)}"
    return f"Use the {row.tool_name} tool."
