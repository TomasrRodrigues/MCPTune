"""Format contract: emit -> chat template -> parse -> original call.

The contract between the Issue 2 emitter (producer) and the Issue 1
runtime parser (consumer). If this passes, a model trained on this data
emits calls the runtime can parse. Needs the real Qwen tokenizer (small
download, no weights); skips cleanly if transformers/network absent.
"""

import pytest

from mcptune.formats.tool_use import ToolUseFormat
from mcptune.runtime.parsing import ToolCall, parse_tool_calls
from mcptune.schema.dataset import DatasetRow
from mcptune.schema.tools import ToolParameter, ToolSpec

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.mark.integration
def test_emit_render_parse_roundtrip():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:  # offline / not cached
        pytest.skip(f"Qwen tokenizer unavailable: {e}")

    tool = ToolSpec(
        "get_weather",
        "Get the weather for a city.",
        [ToolParameter("city", {"type": "string"}, True, "")],
        raw_input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    row = DatasetRow(
        tool_name="get_weather",
        arguments={"city": "Lisbon"},
        request={"id": "x"},
        user_intent="What's the weather in Lisbon?",
    )

    emitted = ToolUseFormat().format_tool([row], [tool])[0]
    text = tokenizer.apply_chat_template(
        emitted["messages"],
        tools=emitted["tools"],
        tokenize=False,
        add_generation_prompt=False,
    )

    assert ToolCall("get_weather", {"city": "Lisbon"}) in parse_tool_calls(text)
