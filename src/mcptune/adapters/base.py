from abc import ABC, abstractmethod

from mcptune.schema.tools import ToolSpec


class MCPAdapter(ABC):
    """
    Abstract interface for MCP (Model Context Protocol) transport adapters.

    MCPAdapter defines the minimal contract required for MCPTune to:
    - discover available tools from an MCP server
    - execute tool calls against that server

    This abstraction allows MCPTune to remain transport-agnostic:
    implementations can use HTTP, SSE, stdio, or any custom protocol
    without changing the dataset or training logic.
    """

    @abstractmethod
    async def discover_tools(self) -> list[ToolSpec]:
        """
        Retrieve all tools exposed by the MCP server.

        Returns
        -------
        list[ToolSpec]
            A structured list of tool definitions including:
            - tool name
            - description
            - parameter schemas
            - optional metadata

        Notes
        -----
        This method is expected to be called once per session
        during dataset construction or tool inspection.
        """
        pass



    @abstractmethod
    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """
        Execute a tool call against the MCP server.

        Parameters
        ----------
        tool_name:
            Name of the tool to invoke.

        arguments:
            Dictionary of validated arguments matching the tool schema.

        Returns
        -------
        dict
            Raw response from the MCP server. The structure is
            backend-specific but typically includes:
            - result data
            - metadata
            - error information (if applicable)

        Notes
        -----
        MCPTune does NOT assume:
        - response format
        - latency characteristics
        - execution semantics

        All interpretation of results happens in higher-level layers.
        """
        pass