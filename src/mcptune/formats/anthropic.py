"""Anthropic Messages API tool_use format.

Three turns per row: user -> assistant (with a tool_use content block)
-> user (with a tool_result content block). Note Anthropic puts
tool_result back in a user turn — there is no separate tool role.
"""

import json

from mcptune.schema.dataset import DatasetRow

from ._common import template_intent


def anthropic_tool_use(rows: list[DatasetRow]) -> list[dict]:
    output = []
    for row in rows:
        user_message = row.user_intent or template_intent(row)
        call_id = row.request["id"]
        tool_result_content = json.dumps(row.response) if row.response is not None else ""

        output.append(
            {
                "messages": [
                    {"role": "user", "content": user_message},
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": call_id,
                                "name": row.tool_name,
                                "input": row.arguments,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": call_id,
                                "content": tool_result_content,
                            }
                        ],
                    },
                ]
            }
        )
    return output
