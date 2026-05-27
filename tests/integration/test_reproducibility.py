import pytest

from mcptune import MCPTune
from mcptune.schema import ToolParameter, ToolSpec


@pytest.mark.unit
def test_dataset_reproducibility_same_seed():
    seed = 42

    mcp1 = MCPTune(model="test", mcpserver=None, seed=seed)
    mcp2 = MCPTune(model="test", mcpserver=None, seed=seed)

    tools = [
        ToolSpec(
            name="tool",
            description="test tool",
            parameters=[
                ToolParameter(
                    name="a",
                    schema={"type": "integer", "minimum": 1, "maximum": 100},
                    required=True,
                    description="a",
                ),
                ToolParameter(
                    name="b",
                    schema={"type": "string", "minLength": 5, "maxLength": 5},
                    required=True,
                    description="b",
                ),
            ],
        )
    ]

    d1 = mcp1.build_dataset(tools)
    d2 = mcp2.build_dataset(tools)

    assert d1 == d2


@pytest.mark.unit
def test_dataset_changes_with_different_seed():
    mcp1 = MCPTune(model="test", mcpserver=None, seed=1)
    mcp2 = MCPTune(model="test", mcpserver=None, seed=2)

    tools = [
        ToolSpec(
            name="tool",
            description="test tool",
            parameters=[
                ToolParameter(
                    name="x",
                    schema={"type": "integer", "minimum": 0, "maximum": 50},
                    required=True,
                    description="x",
                )
            ],
        )
    ]

    d1 = mcp1.build_dataset(tools)
    d2 = mcp2.build_dataset(tools)

    assert d1 != d2


@pytest.mark.unit
def test_request_ids_are_deterministic():
    seed = 123

    mcp1 = MCPTune(model="test", mcpserver=None, seed=seed)
    mcp2 = MCPTune(model="test", mcpserver=None, seed=seed)

    tools = [
        ToolSpec(
            name="tool",
            description="test tool",
            parameters=[
                ToolParameter(
                    name="a",
                    schema={"type": "integer"},
                    required=True,
                    description="a",
                )
            ],
        )
    ]

    d1 = mcp1.build_dataset(tools)
    d2 = mcp2.build_dataset(tools)

    assert d1[0].request["id"] == d2[0].request["id"]


@pytest.mark.unit
def test_tool_order_does_not_affect_output():
    seed = 999

    mcp = MCPTune(model="test", mcpserver=None, seed=seed)

    tools_1 = [
        ToolSpec(
            name="b",
            description="b",
            parameters=[
                ToolParameter(
                    name="x",
                    schema={"type": "integer"},
                    required=True,
                    description="x",
                )
            ],
        ),
        ToolSpec(
            name="a",
            description="a",
            parameters=[
                ToolParameter(
                    name="x",
                    schema={"type": "integer"},
                    required=True,
                    description="x",
                )
            ],
        ),
    ]

    tools_2 = list(reversed(tools_1))

    d1 = mcp.build_dataset(tools_1)
    d2 = mcp.build_dataset(tools_2)

    assert d1 == d2