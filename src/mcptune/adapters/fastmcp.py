from fastmcp import Client

from mcptune.adapters.base import MCPAdapter
from mcptune.schema.tools import ToolParameter, ToolSpec


class FastMCPAdapter(MCPAdapter):
    """Adapter that talks to a FastMCP server through its Client interface.

    The server is wrapped in an in-memory Client, so calls go through the
    same public API that an HTTP or stdio adapter would use - only the
    transport substrate differs.
    """

    def __init__(self, server):
        self.server = server

    async def discover_tools(self) -> list[ToolSpec]:
        async with Client(self.server) as client:
            tools = await client.list_tools()

        return [self._to_toolspec(tool) for tool in tools]

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        async with Client(self.server) as client:
            result = await client.call_tool(tool_name, arguments)

        return self._normalize_response(result)

    def _to_toolspec(self, tool) -> ToolSpec:
        """Convert an MCP Tool object into our internal ToolSpec."""
        schema = tool.inputSchema or {}
        props = schema.get("properties", {})
        required = schema.get("required", [])

        parameters = [
            ToolParameter(
                name=name,
                schema=props[name],
                required=name in required,
                description=props[name].get("description", ""),
            )
            for name in props
        ]

        return ToolSpec(
            name=tool.name,
            description=tool.description or "",
            parameters=parameters,
            raw_input_schema=schema,
        )

    def _normalize_response(self, result) -> dict:
        """Convert FastMCP's CallToolResult into a transport-agnostic dict."""
        return {
            "content": [block.model_dump() for block in result.content],
            "structured_content": result.structured_content,
            "is_error": result.is_error,
        }
