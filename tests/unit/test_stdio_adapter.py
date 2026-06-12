from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest

from mcptune.adapters.stdio import StdioAdapter


# --- construction (no mocking needed) --------------------------------------

def test_defaults():
    a = StdioAdapter(server_path="/srv.py")
    assert a.server_path == "/srv.py"
    assert a.env == {}

def test_env_stored():
    a = StdioAdapter(server_path="/srv.py", env={"K": "v"})
    assert a.env == {"K": "v"}


# --- _client builds the right transport ------------------------------------

@patch("mcptune.adapters.stdio.PythonStdioTransport")
@patch("mcptune.adapters.stdio.Client")
def test_client_builds_stdio_transport_with_env(mock_client, mock_transport):
    a = StdioAdapter(server_path="/srv.py", env={"API_KEY": "x"})
    a._client()
    mock_transport.assert_called_once_with("/srv.py", env={"API_KEY": "x"})
    mock_client.assert_called_once_with(mock_transport.return_value)


# --- discover_tools normalizes each tool -----------------------------------

async def test_discover_tools_normalizes(monkeypatch):
    raw_tool = MagicMock()
    raw_tool.name = "add"
    raw_tool.description = "Add"
    raw_tool.inputSchema = {"properties": {"a": {"type": "integer"}}, "required": ["a"]}

    fake = MagicMock()
    fake.list_tools = _async_return([raw_tool])

    a = StdioAdapter(server_path="/srv.py")
    monkeypatch.setattr(a, "_client", lambda: _ctx(fake))

    tools = await a.discover_tools()
    assert len(tools) == 1
    assert tools[0].name == "add"
    assert tools[0].raw_input_schema["required"] == ["a"]


# --- call_tool passes raise_on_error=False AND normalizes ------------------

async def test_call_tool_passes_raise_on_error_false(monkeypatch):
    from unittest.mock import AsyncMock

    result_obj = MagicMock()
    result_obj.content = []
    result_obj.structured_content = {"result": 5}
    result_obj.is_error = False

    fake = MagicMock()
    fake.call_tool = AsyncMock(return_value=result_obj)   # records call args

    a = StdioAdapter(server_path="/srv.py")
    monkeypatch.setattr(a, "_client", lambda: _ctx(fake))

    out = await a.call_tool("add", {"a": 2, "b": 3})

    # the contract: raise_on_error=False was passed through
    fake.call_tool.assert_awaited_once_with("add", {"a": 2, "b": 3}, raise_on_error=False)
    # and the response was normalized to the shared shape
    assert set(out.keys()) == {"content", "structured_content", "is_error"}
    assert out["is_error"] is False
    assert out["structured_content"]["result"] == 5


# --- helpers ---------------------------------------------------------------

def _async_return(value):
    async def _fn(*args, **kwargs):
        return value
    return _fn

@asynccontextmanager
async def _ctx(client):
    yield client