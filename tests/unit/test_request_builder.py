import pytest

from mcptune import MCPTune
from mcptune.schema.tools import ToolSpec


@pytest.mark.unit
def test_request_envelope_is_jsonrpc_2():
    m = MCPTune(model="x", mcpserver=None)
    tool = ToolSpec(name="weather", description="", parameters=[])

    req = m.build_mcp_request(tool, {"city": "Porto"})

    assert req["jsonrpc"] == "2.0"
    assert req["method"] == "tools/call"


@pytest.mark.unit
def test_request_carries_tool_name_and_arguments():
    m = MCPTune(model="x", mcpserver=None)
    tool = ToolSpec(name="weather", description="", parameters=[])

    req = m.build_mcp_request(tool, {"city": "Porto"})

    assert req["params"]["name"] == "weather"
    assert req["params"]["arguments"] == {"city": "Porto"}


@pytest.mark.unit
def test_request_has_unique_id_per_call():
    m = MCPTune(model="x", mcpserver=None)
    tool = ToolSpec(name="weather", description="", parameters=[])

    assert m.build_mcp_request(tool, {})["id"] != m.build_mcp_request(tool, {})["id"]


@pytest.mark.unit
def test_seeded_requests_are_reproducible_across_instances():
    tool = ToolSpec(name="weather", description="", parameters=[])
    first = MCPTune(model="x", mcpserver=None, seed=42)
    second = MCPTune(model="x", mcpserver=None, seed=42)

    assert first.build_mcp_request(tool, {}) == second.build_mcp_request(tool, {})
