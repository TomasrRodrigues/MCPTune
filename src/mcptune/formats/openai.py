"""OpenAI Chat Completions format with tool_calls.

Each DatasetRow becomes a single three-message conversation:
user -> assistant (one tool_call) -> tool (the call's response).
"""

import json

from mcptune.schema.dataset import DatasetRow


def openai_messages(rows: list[DatasetRow]) -> list[dict]:
    output = []
    for row in rows:
        user_message = row.user_intent or _template_intent(row)
        call_id = row.request["id"]
        tool_content = json.dumps(row.response) if row.response is not None else ""

        output.append(
            {
                "messages": [
                    {"role": "user", "content": user_message},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": row.tool_name,
                                    "arguments": json.dumps(row.arguments),
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": tool_content,
                    },
                ]
            }
        )
    return output


def _template_intent(row: DatasetRow) -> str:
    """Fallback for DatasetRows constructed without user_intent. Rows from
    MCPTune.build_dataset always have one; this fires only for manual rows."""
    if row.arguments:
        return f"Use the {row.tool_name} tool with arguments: {json.dumps(row.arguments)}"
    return f"Use the {row.tool_name} tool."
