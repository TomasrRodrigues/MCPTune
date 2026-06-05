"""OpenAI Chat Completions format with tool_calls.

Each DatasetRow becomes a single three-message conversation:
user -> assistant (one tool_call) -> tool (the call's response).
"""

import json
from typing import Any

from mcptune.formats.base import Format
from mcptune.schema.dataset import DatasetRow

from ..schema.tools import ToolSpec
from ._common import template_intent


class OpenAIFormat(Format):
    def format_tool(
        self, rows: list[DatasetRow], tools: list[ToolSpec] | None = None
    ) -> list[dict[str, Any]]:
        output: list[dict[str, Any]] = []
        for row in rows:
            user_message = row.user_intent or template_intent(row)
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
