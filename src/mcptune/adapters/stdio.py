from typing import Any

from fastmcp import Client
from fastmcp.client.transports import PythonStdioTransport

from mcptune.adapters.base import MCPAdapter
from mcptune.schema.tools import ToolSpec


class StdioAdapter(MCPAdapter):
    """MCPAdapter for MCP servers shipped as stdio subprocess binaries.

    Targets the most common real-world deployment (Claude Desktop and most
    third-party servers): a local Python server spoken to over stdin/stdout.
    FastMCP launches and manages the subprocess; this adapter supplies the
    server path and optional environment (API keys, etc.).

    v1 lifecycle: a fresh subprocess is spawned per discover/call, because a
    Client is opened per operation. Wasteful but simple; persistent session
    reuse is a later optimization (see docs/adapters.md).

    Error model:
        - A tool that returns/raises an error is a normal result with
          is_error=True (via the shared _normalize_response).
        - A subprocess that cannot start or dies is NOT a tool result; the
          underlying exception propagates so callers can record it on
          DatasetRow.error.
    """

    def __init__(self, server_path: str, env: dict[str, str] | None = None):
        """
        Parameters
            server_path:
                Path to the Python MCP server script to launch.
            env:
                Optional environment variables for the subprocess (e.g. API
                keys the server requires). Merged by FastMCP into the child
                process environment.
        """
        self.server_path = server_path
        self.env = env or {}

    def _client(self) -> Client:
        transport = PythonStdioTransport(self.server_path, env=self.env)
        return Client(transport)

    async def discover_tools(self) -> list[ToolSpec]:
        async with self._client() as client:
            tools = await client.list_tools()
        return [self._to_toolspec(tool) for tool in tools]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with self._client() as client:
            result = await client.call_tool(tool_name, arguments, raise_on_error=False)
        return self._normalize_response(result)
