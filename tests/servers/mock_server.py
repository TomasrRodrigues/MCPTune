# tests/mcp/mock_server.py

from typing import Any


class MockMCPServer:
    def __init__(self) -> None:
        self.tools: dict[str, Any] = {
            "add": self.add,
            "echo": self.echo,
        }

    def list_tools(self) -> list[str]:
        return list(self.tools.keys())

    def call_tool(self, name: str, args: dict[str, Any]) -> Any:
        return self.tools[name](**args)

    def add(self, a: int, b: int) -> int:
        return a + b

    def echo(self, text: str) -> str:
        return text
