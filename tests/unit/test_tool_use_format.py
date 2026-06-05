import pytest

from mcptune.formats.tool_use import ToolUseFormat
from mcptune.schema.dataset import DatasetRow
from mcptune.schema.tools import ToolParameter, ToolSpec


def _tools():
    return [
        ToolSpec(
            "get_weather",
            "Get the weather for a city.",
            [ToolParameter("city", {"type": "string"}, True, "")],
            raw_input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        ),
        ToolSpec(
            "send_email",
            "Send an email.",
            [ToolParameter("to", {"type": "string"}, True, "")],
            raw_input_schema={
                "type": "object",
                "properties": {"to": {"type": "string"}},
                "required": ["to"],
            },
        ),
    ]


def _row():
    return DatasetRow(
        tool_name="get_weather",
        arguments={"city": "Lisbon"},
        request={"id": "x"},
        user_intent="What's the weather in Lisbon?",
    )


def test_includes_full_tool_surface_not_just_called_tool():
    out = ToolUseFormat().format_tool([_row()], _tools())
    names = {t["function"]["name"] for t in out[0]["tools"]}
    assert names == {"get_weather", "send_email"}


def test_assistant_call_is_structured_not_stringified():
    assistant = ToolUseFormat().format_tool([_row()], _tools())[0]["messages"][1]
    assert assistant.get("content") in (None, "")  # not stringified into content
    call = assistant["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "get_weather"
    assert call["function"]["arguments"] == {"city": "Lisbon"}  # dict, not a JSON string


def test_user_turn_uses_intent():
    out = ToolUseFormat().format_tool([_row()], _tools())
    assert out[0]["messages"][0] == {"role": "user", "content": "What's the weather in Lisbon?"}


def test_requires_tools():
    with pytest.raises(ValueError):
        ToolUseFormat().format_tool([_row()], [])
