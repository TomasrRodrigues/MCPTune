from abc import ABC, abstractmethod


class Backend(ABC):
    """
    Executes MCP tool calls.
    """

    @abstractmethod
    def call_tool(self, tool_name: str, arguments: dict) -> dict:
        pass
