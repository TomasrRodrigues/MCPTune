"""Native tool-use training format (Gaps A + B).

Unlike the other emitters, this produces data a runtime can consume:

  A) the available tool definitions are included in context (`tools`),
     so the model learns to condition on what's available and pick the
     right tool — not just transcribe a named one;
  B) the assistant turn carries a STRUCTURED tool call, not stringified
     JSON in content. The native <tool_call> text is produced by the
     tokenizer's chat template at training time (apply_chat_template
     with tools=), the SAME rendering used at inference. That identity
     is what makes the trained model's output parseable by the runtime.

We deliberately do NOT hand-write <tool_call> strings. Emitting the
structured form and letting the chat template render it guarantees
training text == inference text, and lets one emitter serve any model
family whose template supports tools= (Qwen first).

Each row becomes a two-turn example: user intent -> assistant tool call.
The tool result and final synthesizing answer are added by Issue 3.
"""

from __future__ import annotations

from ..schema.dataset import DatasetRow
from ..schema.function_schema import toolspecs_to_function_schemas
from ..schema.tools import ToolSpec
from ._common import template_intent


def tool_use(rows: list[DatasetRow], tools: list[ToolSpec]) -> list[dict]:
    """Emit native tool-use training rows.

    `tools` is the FULL available tool surface — rendered into every
    example so the model learns selection, not just the tools that
    happen to appear in `rows`.

    Returns items shaped {"messages": [...], "tools": [...schemas...]};
    pass both into tokenizer.apply_chat_template(messages, tools=tools).
    """
    if not tools:
        raise ValueError(
            "tool_use format requires the available tools to render into "
            "context (Gap A). Pass the ToolSpec list from discover()."
        )

    function_schemas = toolspecs_to_function_schemas(tools)
    output: list[dict] = []

    for row in rows:
        user_message = row.user_intent or template_intent(row)
        output.append(
            {
                "messages": [
                    {"role": "user", "content": user_message},
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": row.tool_name,
                                    "arguments": row.arguments,  # dict; template renders it
                                },
                            }
                        ],
                    },
                ],
                "tools": function_schemas,
            }
        )

    return output
