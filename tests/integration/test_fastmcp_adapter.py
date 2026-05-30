"""Baseline integration tests for the FastMCP adapter.

Covers the discovery → schema-normalization → ToolSpec pipeline. When
additional transport adapters land (Issue 8: HTTP, stdio), the fixture
and the assertions here are intended to be promoted into a parametrized
contract test that runs against every adapter - anything we assert here
is part of the adapter interface, regardless of transport.
"""

import pytest
from fastmcp import FastMCP

from mcptune.adapters.fastmcp import FastMCPAdapter
from mcptune.schema.tools import ToolParameter, ToolSpec

# ---------------------------------------------------------------------------
# Fixture: a FastMCP server with deliberately diverse tool shapes.
# ---------------------------------------------------------------------------


@pytest.fixture
def rich_fastmcp_server() -> FastMCP:
    """Server whose tools cover the normalization edge cases:
    required-only, optional+required, no parameters, multiple types."""
    mcp = FastMCP("adapter-test-server")

    @mcp.tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Weather for {city}"

    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @mcp.tool
    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone with an optional custom greeting."""
        return f"{greeting}, {name}"

    @mcp.tool
    def ping() -> str:
        """Health check - takes no arguments."""
        return "pong"

    @mcp.tool
    def divide(a: float, b: float) -> float:
        """Divide two floating-point numbers."""
        return a / b

    return mcp


@pytest.fixture
def rich_adapter(rich_fastmcp_server) -> FastMCPAdapter:
    return FastMCPAdapter(rich_fastmcp_server)


# ---------------------------------------------------------------------------
# Discovery contract
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_discover_returns_all_registered_tools(rich_adapter):
    tools = await rich_adapter.discover_tools()
    names = {t.name for t in tools}
    assert names == {"get_weather", "add", "greet", "ping", "divide"}


@pytest.mark.integration
async def test_discover_returns_toolspec_instances(rich_adapter):
    tools = await rich_adapter.discover_tools()
    assert all(isinstance(t, ToolSpec) for t in tools)


@pytest.mark.integration
async def test_every_tool_has_a_description(rich_adapter):
    tools = await rich_adapter.discover_tools()
    for tool in tools:
        assert tool.description, f"{tool.name} has empty description"


@pytest.mark.integration
async def test_tool_with_no_parameters_yields_empty_parameter_list(rich_adapter):
    tools = await rich_adapter.discover_tools()
    ping = next(t for t in tools if t.name == "ping")
    assert ping.parameters == []


# ---------------------------------------------------------------------------
# Schema fidelity - the raw JSONSchema must survive normalization intact
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_raw_input_schema_is_preserved(rich_adapter):
    tools = await rich_adapter.discover_tools()
    weather = next(t for t in tools if t.name == "get_weather")
    assert weather.raw_input_schema is not None
    assert "properties" in weather.raw_input_schema


@pytest.mark.integration
async def test_raw_input_schema_preserves_required_list(rich_adapter):
    tools = await rich_adapter.discover_tools()
    add = next(t for t in tools if t.name == "add")
    assert set(add.raw_input_schema["required"]) == {"a", "b"}


# ---------------------------------------------------------------------------
# Parameter normalization
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_single_required_string_parameter_is_normalized(rich_adapter):
    tools = await rich_adapter.discover_tools()
    weather = next(t for t in tools if t.name == "get_weather")

    assert len(weather.parameters) == 1
    param = weather.parameters[0]
    assert isinstance(param, ToolParameter)
    assert param.name == "city"
    assert param.schema["type"] == "string"
    assert param.required is True


@pytest.mark.integration
async def test_multiple_required_parameters_are_normalized(rich_adapter):
    tools = await rich_adapter.discover_tools()
    add = next(t for t in tools if t.name == "add")

    assert {p.name for p in add.parameters} == {"a", "b"}
    assert all(p.required for p in add.parameters)
    assert all(p.schema["type"] == "integer" for p in add.parameters)


@pytest.mark.integration
async def test_optional_parameters_are_distinguished_from_required(rich_adapter):
    tools = await rich_adapter.discover_tools()
    greet = next(t for t in tools if t.name == "greet")

    by_name = {p.name: p for p in greet.parameters}
    assert by_name["name"].required is True
    assert by_name["greeting"].required is False


@pytest.mark.integration
async def test_parameter_type_is_preserved_for_floats(rich_adapter):
    tools = await rich_adapter.discover_tools()
    divide = next(t for t in tools if t.name == "divide")
    assert all(p.schema["type"] == "number" for p in divide.parameters)


# ---------------------------------------------------------------------------
# Execution behaviour - smoke checks at the adapter level.
# Exhaustive execution / error-path tests belong with execute_dataset.
# ---------------------------------------------------------------------------


@pytest.mark.integration
async def test_call_tool_returns_normalized_contract(rich_adapter):
    result = await rich_adapter.call_tool("add", {"a": 2, "b": 3})
    assert set(result.keys()) == {"content", "structured_content", "is_error"}


@pytest.mark.integration
async def test_call_tool_success_marks_is_error_false(rich_adapter):
    result = await rich_adapter.call_tool("add", {"a": 2, "b": 3})
    assert result["is_error"] is False
    assert result["structured_content"]["result"] == 5
