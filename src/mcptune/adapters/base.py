# mcptune/adapters/base.py
from abc import ABC, abstractmethod
from typing import Any

from mcptune.schema.tools import ToolParameter, ToolSpec


class MCPAdapter(ABC):
    """Abstract interface for MCP transport adapters.

    Subclasses implement transport (how to obtain a connected FastMCP
    ``Client``); normalization of tools and responses is shared here so
    every adapter produces an identical ToolSpec / response shape.
    """

    @abstractmethod
    async def discover_tools(self) -> list[ToolSpec]:
        """Retrieve all tools exposed by the MCP server as ToolSpecs."""
        ...

    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: dict[Any, Any]) -> dict[str, Any]:
        """Execute a tool call and return the normalized response dict."""
        ...

    # shared normalization (transport-agnostic)

    def _to_toolspec(self, tool: Any) -> ToolSpec:
        """Convert a FastMCP tool object into MCPTune's ToolSpec format."""
        schema: Any = tool.inputSchema or {}
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

    def _normalize_response(self, result: Any) -> dict[str, Any]:
        """Normalize a FastMCP CallToolResult into a transport-agnostic dict.

        Output shape is fixed: ``content``, ``structured_content``,
        ``is_error``. Every adapter returns exactly these keys.
        """
        return {
            "content": [block.model_dump() for block in result.content],
            "structured_content": result.structured_content,
            "is_error": result.is_error,
        }
