import pytest

from mcptune.runtime.parsing import ToolCall, parse_tool_calls
from mcptune.schema.function_schema import toolspec_to_function_schema
from mcptune.schema.tools import ToolParameter, ToolSpec


def test_parses_single_qwen_call():
    text = '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Lisbon"}}\n</tool_call>'
    assert parse_tool_calls(text) == [ToolCall("get_weather", {"city": "Lisbon"})]


def test_parses_multiple_calls():
    text = (
        '<tool_call>{"name": "a", "arguments": {}}</tool_call>'
        '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>'
    )
    calls = parse_tool_calls(text)
    assert [c.name for c in calls] == ["a", "b"]
    assert calls[1].arguments == {"x": 1}


def test_no_call_is_final_answer():
    assert parse_tool_calls("It's sunny in Lisbon today.") == []


def test_malformed_json_skipped():
    assert parse_tool_calls("<tool_call>{nope}</tool_call>") == []


def test_unknown_format_raises():
    with pytest.raises(ValueError):
        parse_tool_calls("x", fmt="llama")


def test_function_schema_preserves_raw_schema():
    raw = {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}
    tool = ToolSpec(
        name="get_weather",
        description="Get weather",
        parameters=[ToolParameter("city", {"type": "string"}, True, "")],
        raw_input_schema=raw,
    )
    fn = toolspec_to_function_schema(tool)
    assert fn["function"]["name"] == "get_weather"
    assert fn["function"]["parameters"] == raw
