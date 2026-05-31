from fastmcp import Client

from mcptune.adapters.base import MCPAdapter
from mcptune.schema.tools import ToolParameter, ToolSpec


class FastMCPAdapter(MCPAdapter):
    """
    FastMCP implementation of MCPAdapter.

    This adapter connects MCPTune to a FastMCP server using the official
    FastMCP Client interface.

    It acts as a thin translation layer between:
    - FastMCP native tool representations
    - MCPTune internal ToolSpec / DatasetRow schema

    No sampling or dataset logic is performed here.
    This class is purely responsible for transport + normalization.
    """

    def __init__(self, server):
        """
        Parameters
        ----------
        server:
            FastMCP server instance or connection target.
            Passed directly into FastMCP Client.
        """
        self.server = server

    async def discover_tools(self) -> list[ToolSpec]:
        """
        Fetch tool definitions from the MCP server and convert them
        into internal ToolSpec objects.

        Returns
        -------
        list[ToolSpec]
            Normalized tool specifications used by MCPTune.
        """
        async with Client(self.server) as client:
            tools = await client.list_tools()

        return [self._to_toolspec(tool) for tool in tools]

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Execute a tool call on the MCP server.

        Parameters
        ----------
        tool_name:
            Name of the tool to invoke.

        arguments:
            Dictionary of arguments matching the tool schema.

        Returns
        -------
        dict
            Normalized response with:
            - content blocks
            - structured content (if available)
            - error flag

        Notes
        -----
        This is a thin wrapper over FastMCP's call_tool API.
        No semantic interpretation is performed here.
        """
        async with Client(self.server) as client:
            result = await client.call_tool(tool_name, arguments)

        return self._normalize_response(result)

    def _to_toolspec(self, tool) -> ToolSpec:
        """
        Convert a FastMCP tool object into MCPTune's ToolSpec format.

        This step normalizes schema representation so that all downstream
        components (samplers, intent models, trainers) operate on a
        consistent structure.
        """
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
        """
        Normalize FastMCP CallToolResult into a transport-agnostic dict.

        This ensures MCPTune never depends on FastMCP-specific structures.

        Output format is intentionally minimal:
        - content: raw response blocks
        - structured_content: optional parsed payload
        - is_error: execution status flag
        """
        return {
            "content": [block.model_dump() for block in result.content],
            "structured_content": result.structured_content,
            "is_error": result.is_error,
        }
