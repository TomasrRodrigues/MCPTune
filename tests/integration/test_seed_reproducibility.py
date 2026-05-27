import pytest

from mcptune import MCPTune
from mcptune.schema import ToolParameter, ToolSpec


@pytest.mark.unit
def test_same_seed_produces_identical_dataset():
    tools = [
        ToolSpec(
            name="tool_a",
            description="a",
            parameters=[
                ToolParameter(
                    name="x",
                    schema={"type": "integer"},
                    required=True,
                    description="x",
                )
            ],
        )
    ]

    mcp1 = MCPTune(model="test", mcpserver=None, seed=123)
    mcp2 = MCPTune(model="test", mcpserver=None, seed=123)

    d1 = mcp1.build_dataset(tools)
    d2 = mcp2.build_dataset(tools)

    assert d1 == d2


@pytest.mark.unit
def test_different_seed_produces_different_dataset():
    tools = [
        ToolSpec(
            name="tool_a",
            description="a",
            parameters=[
                ToolParameter(
                    name="x",
                    schema={"type": "integer"},
                    required=True,
                    description="x",
                )
            ],
        )
    ]

    mcp1 = MCPTune(model="test", mcpserver=None, seed=1)
    mcp2 = MCPTune(model="test", mcpserver=None, seed=2)

    d1 = mcp1.build_dataset(tools)
    d2 = mcp2.build_dataset(tools)

    assert d1 != d2