# mcptune/adapters/http.py
from typing import Any

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport

from mcptune.adapters.base import MCPAdapter
from mcptune.schema.tools import ToolSpec


class HTTPAdapter(MCPAdapter):
    """MCPAdapter for MCP servers deployed as HTTP services.

    Targets production/remote deployments reachable over HTTP. FastMCP
    selects the concrete transport (streamable-http or SSE) from the URL;
    this adapter only supplies the URL, optional static auth headers, and
    a per-call timeout.

    Auth in v1 is "pass a header dict" (e.g. ``{"Authorization": "Bearer ..."}``).
    OAuth/refresh flows, retries, and backoff are intentionally out of scope.
    Call failures are not caught here - they propagate so the caller can
    record them on ``DatasetRow.error``.
    """

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    )-> None:
        """
        Parameters
            url:      Base URL of the HTTP-deployed MCP server.
            headers:  Optional static headers sent on every request
                      (auth, tenant routing, etc.).
            timeout:  Per-call timeout in seconds. Default 30.
        """
        self.url = url
        self.headers = headers or {}
        self.timeout = timeout

    def _client(self) -> Client:
        """Build a FastMCP Client over an explicit HTTP transport.

        Using StreamableHttpTransport (rather than passing the bare URL)
        is what lets us attach headers; the transport is the only place
        FastMCP exposes per-request header injection.
        """
        transport = StreamableHttpTransport(url=self.url, headers=self.headers)
        return Client(transport, timeout=self.timeout)

    async def discover_tools(self) -> list[ToolSpec]:
        async with self._client() as client:
            tools = await client.list_tools()
        return [self._to_toolspec(tool) for tool in tools]

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        async with self._client() as client:
            result = await client.call_tool(tool_name, arguments)
        return self._normalize_response(result)