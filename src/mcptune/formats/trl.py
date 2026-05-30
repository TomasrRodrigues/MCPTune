"""HuggingFace TRL SFTTrainer format.

{messages: [{role, content}]} with stringified content for chat templates
that don't natively handle structured tool calls. For tool-aware chat
templates (e.g., Qwen2's), use the openai format directly with TRL.
"""

import json

from mcptune.schema.dataset import DatasetRow

from ._common import template_intent


def trl_sft(rows: list[DatasetRow]) -> list[dict]:
    output = []
    for row in rows:
        user_message = row.user_intent or template_intent(row)
        assistant_message = json.dumps(
            {
                "name": row.tool_name,
                "arguments": row.arguments,
            }
        )
        tool_message = json.dumps(row.response) if row.response is not None else ""

        output.append(
            {
                "messages": [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": assistant_message},
                    {"role": "tool", "content": tool_message},
                ]
            }
        )
    return output
