"""E2E tests for StdioAdapter against a real spawned stdio MCP server.

Opt-in: marked e2e, excluded from the default run by addopts = "-m 'not e2e'".
Run with:  pytest -m e2e
"""

from pathlib import Path

import pytest

from mcptune.adapters.stdio import StdioAdapter
from mcptune.schema.tools import ToolSpec

SERVER = str(Path(__file__).parent / "servers" / "mcp_stdio_server.py")


@pytest.fixture
def adapter():
    return StdioAdapter(server_path=SERVER)


# --- discovery -------------------------------------------------------------


@pytest.mark.e2e
async def test_discover_returns_toolspecs(adapter):
    tools = await adapter.discover_tools()
    assert all(isinstance(t, ToolSpec) for t in tools)
    assert {"add", "boom", "echo_env"} <= {t.name for t in tools}


@pytest.mark.e2e
async def test_discover_preserves_schema(adapter):
    tools = {t.name: t for t in await adapter.discover_tools()}
    add = tools["add"].raw_input_schema
    assert add["properties"]["a"]["type"] == "integer"
    assert set(add.get("required", [])) == {"a", "b"}


# --- normalized contract (identical shape to the other two adapters) -------


@pytest.mark.e2e
async def test_call_tool_normalized_contract(adapter):
    result = await adapter.call_tool("add", {"a": 2, "b": 3})
    assert set(result.keys()) == {"content", "structured_content", "is_error"}
    assert result["is_error"] is False
    assert result["structured_content"]["result"] == 5


# --- the crash vs. tool-error distinction (the heart of this issue) --------


@pytest.mark.e2e
async def test_tool_error_returns_is_error_true(adapter):
    """A tool that raises is a NORMAL result with is_error=True - not a crash."""
    result = await adapter.call_tool("boom", {})
    assert result["is_error"] is True
    assert result["content"]  # error detail preserved, not swallowed


@pytest.mark.e2e
async def test_subprocess_crash_raises(self_path="/does/not/exist/server.py"):
    """A bad server path = the subprocess can't start = this RAISES, it does
    not return is_error. Crashes propagate so they land on DatasetRow.error."""
    bad = StdioAdapter(server_path=self_path)
    with pytest.raises(FileNotFoundError):
        await bad.discover_tools()


# env injection


@pytest.mark.e2e
async def test_env_is_passed_to_subprocess():
    """env dict at construction must reach the spawned process."""
    adapter = StdioAdapter(server_path=SERVER, env={"MCPTUNE_TEST_KEY": "secret123"})
    result = await adapter.call_tool("echo_env", {"key": "MCPTUNE_TEST_KEY"})
    assert "secret123" in str(result["structured_content"]) + str(result["content"])




# --- defaults (no subprocess) ----------------------------------------------


def test_defaults():
    a = StdioAdapter(server_path=SERVER)
    assert a.env == {}
    assert a.server_path == SERVER


