# tests/integration/test_http_adapter.py
"""Integration tests for HTTPAdapter against a live in-process HTTP MCP server.

A FastMCP server is mounted as an ASGI app and served by uvicorn on a
background thread bound to an ephemeral port. The adapter then talks to it
over real HTTP (localhost), exercising the streamable-http transport rather
than the in-memory shortcut. Assertions mirror the FastMCP adapter's
normalized-contract tests so both transports are proven to produce the
identical shape.
"""

import socket
import threading
import time

import httpx
import pytest
import uvicorn
from fastmcp import FastMCP

from mcptune.adapters.http import HTTPAdapter
from mcptune.schema.tools import ToolSpec


# --- server under test -----------------------------------------------------

def _build_server() -> FastMCP:
    server = FastMCP("http-test")

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @server.tool
    def greet(name: str) -> str:
        """Return a greeting for the given name."""
        return f"Hello, {name}!"

    return server


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def http_url():
    """Serve the FastMCP app over HTTP on a background thread.

    Yields the MCP endpoint URL. The server is shut down when the module's
    tests finish.
    """
    port = _free_port()
    # FastMCP exposes an ASGI app for HTTP transport. http_app() is the
    # streamable-http app in FastMCP 3.x; if your version names it
    # differently (e.g. streamable_http_app()), adjust this one call.
    app = _build_server().http_app()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    base = f"http://127.0.0.1:{port}"
    # FastMCP's streamable-http endpoint is mounted at /mcp/ by default.
    url = f"{base}/mcp/"

    # wait for readiness instead of sleeping a fixed amount
    deadline = time.time() + 10
    while time.time() < deadline:
        if server.started:
            try:
                httpx.get(base, timeout=0.5)
                break
            except httpx.HTTPError:
                break  # connection refused/4xx both mean the port is live
        time.sleep(0.05)
    else:
        raise RuntimeError("HTTP MCP server did not start in time")

    yield url

    server.should_exit = True
    thread.join(timeout=5)


@pytest.fixture
def adapter(http_url):
    return HTTPAdapter(url=http_url)


# --- discovery -------------------------------------------------------------

@pytest.mark.integration
async def test_discover_returns_toolspecs(adapter):
    tools = await adapter.discover_tools()
    assert all(isinstance(t, ToolSpec) for t in tools)
    assert {t.name for t in tools} == {"add", "greet"}


@pytest.mark.integration
async def test_discover_preserves_raw_schema(adapter):
    tools = {t.name: t for t in await adapter.discover_tools()}
    add_schema = tools["add"].raw_input_schema
    assert add_schema["properties"]["a"]["type"] == "integer"
    assert set(add_schema.get("required", [])) == {"a", "b"}


@pytest.mark.integration
async def test_discover_normalizes_parameters(adapter):
    tools = {t.name: t for t in await adapter.discover_tools()}
    params = {p.name: p for p in tools["greet"].parameters}
    assert "name" in params
    assert params["name"].required is True


# --- call_tool: the normalized-contract assertions (mirror FastMCP tests) --

@pytest.mark.integration
async def test_call_tool_returns_normalized_contract(adapter):
    result = await adapter.call_tool("add", {"a": 2, "b": 3})
    assert set(result.keys()) == {"content", "structured_content", "is_error"}


@pytest.mark.integration
async def test_call_tool_success_marks_is_error_false(adapter):
    result = await adapter.call_tool("add", {"a": 2, "b": 3})
    assert result["is_error"] is False
    assert result["structured_content"]["result"] == 5


@pytest.mark.integration
async def test_call_tool_string_result(adapter):
    result = await adapter.call_tool("greet", {"name": "Tomas"})
    assert result["is_error"] is False
    assert "Hello, Tomas!" in str(result["structured_content"]) + str(result["content"])


# --- config plumbing -------------------------------------------------------

@pytest.mark.integration
async def test_headers_and_timeout_are_accepted(http_url):
    """A static header dict and custom timeout must not break a normal call.

    (v1 auth is "pass a header dict"; this server doesn't require auth, so we
    only assert the adapter threads them through without error.)
    """
    adapter = HTTPAdapter(
        url=http_url,
        headers={"X-Test-Header": "mcptune"},
        timeout=10.0,
    )
    result = await adapter.call_tool("add", {"a": 10, "b": 5})
    assert result["structured_content"]["result"] == 15


def test_defaults():
    """Construction defaults match the issue spec (no network)."""
    a = HTTPAdapter(url="http://example.invalid/mcp/")
    assert a.timeout == 30.0
    assert a.headers == {}
    assert a.url == "http://example.invalid/mcp/"