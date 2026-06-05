import random
from typing import Any

from .base import Backend


class MockBackend(Backend):
    """
    Deterministic fake MCP server.
    Used for dataset generation.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def call_tool(
        self, tool_name: str, arguments: dict[str, str | dict[Any, Any]]
    ) -> dict[str, str | Any]:
        # deterministic fake response
        return {"tool": tool_name, "data": arguments, "result": f"mock_result_{tool_name}"}
