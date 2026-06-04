import pytest
from fastmcp import FastMCP

from mcptune.adapters.fastmcp import FastMCPAdapter
from mcptune.runtime import run


class ScriptedRunner:
    """Returns canned outputs in order — stands in for a model."""

    def __init__(self, outputs):
        self.outputs, self.i = list(outputs), 0

    def generate(self, messages, tools):
        out = self.outputs[self.i]
        self.i += 1
        return out


def _server():
    s = FastMCP("t")

    @s.tool
    def get_weather(city: str) -> str:
        return f"Sunny in {city}"

    return s


@pytest.mark.asyncio
async def test_executes_tool_and_feeds_result_back():
    adapter = FastMCPAdapter(_server())
    tools = await adapter.discover_tools()
    runner = ScriptedRunner(
        [
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Lisbon"}}</tool_call>',
            "It's sunny in Lisbon.",
        ]
    )

    result = await run(runner, adapter, "weather in Lisbon?", tools)

    assert result.stopped_reason == "final_answer"
    assert result.final_text == "It's sunny in Lisbon."
    assert [c.name for c in result.tool_calls] == ["get_weather"]
    assert any(m["role"] == "tool" and "Sunny in Lisbon" in m["content"] for m in result.messages)


@pytest.mark.asyncio
async def test_max_turns_cap():
    adapter = FastMCPAdapter(_server())
    tools = await adapter.discover_tools()
    always_calls = ScriptedRunner(
        ['<tool_call>{"name": "get_weather", "arguments": {"city": "X"}}</tool_call>'] * 10
    )
    result = await run(always_calls, adapter, "loop", tools, max_turns=3)
    assert result.stopped_reason == "max_turns"
    assert len(result.tool_calls) == 3


@pytest.mark.asyncio
async def test_unknown_tool_error_fed_back_not_raised():
    adapter = FastMCPAdapter(_server())
    tools = await adapter.discover_tools()
    runner = ScriptedRunner(
        [
            '<tool_call>{"name": "nonexistent", "arguments": {}}</tool_call>',
            "Sorry, I couldn't do that.",
        ]
    )
    result = await run(runner, adapter, "do x", tools)
    assert result.stopped_reason == "final_answer"
    assert any(m["role"] == "tool" and "Error" in m["content"] for m in result.messages)
