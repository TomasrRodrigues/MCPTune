"""ShareGPT-style conversation format.

Three turns per row in {conversations: [{from, value}]} shape, with
roles human / gpt / tool. ShareGPT has no universal tool-calling
convention, so the assistant turn embeds the call as inline JSON.
"""

import json

from mcptune.schema.dataset import DatasetRow

from ._common import template_intent


def sharegpt(rows: list[DatasetRow]) -> list[dict]:
    output = []
    for row in rows:
        user_message = row.user_intent or template_intent(row)
        gpt_value = json.dumps(
            {
                "tool_call": {
                    "name": row.tool_name,
                    "arguments": row.arguments,
                }
            }
        )
        tool_value = json.dumps(row.response) if row.response is not None else ""

        output.append(
            {
                "conversations": [
                    {"from": "human", "value": user_message},
                    {"from": "gpt", "value": gpt_value},
                    {"from": "tool", "value": tool_value},
                ]
            }
        )
    return output
