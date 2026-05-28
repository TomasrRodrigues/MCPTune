import pytest

from mcptune import MCPTune
from mcptune.schema import ToolParameter, ToolSpec


@pytest.mark.unit
def test_dataset_cardinality_matches_samples_per_tool():
    tools = [
        ToolSpec(
            name="tool_a",
            description="a",
            parameters=[
                ToolParameter(name="x", schema={"type": "integer"}, required=True, description="")
            ],
        ),
        ToolSpec(
            name="tool_b",
            description="b",
            parameters=[
                ToolParameter(name="y", schema={"type": "integer"}, required=True, description="")
            ],
        ),
    ]

    mcp = MCPTune(model="test", mcpserver=None, seed=42)

    dataset = mcp.build_dataset(tools, samples_per_tool=5)

    # 2 tools × 5 samples each
    assert len(dataset) == 10


@pytest.mark.unit
def test_each_tool_has_correct_number_of_rows():
    tools = [
        ToolSpec(
            name="tool_a",
            description="a",
            parameters=[
                ToolParameter(name="x", schema={"type": "integer"}, required=True, description="")
            ],
        )
    ]

    mcp = MCPTune(model="test", mcpserver=None, seed=123)

    dataset = mcp.build_dataset(tools, samples_per_tool=7)

    assert len(dataset) == 7
    assert all(row.tool_name == "tool_a" for row in dataset)


@pytest.mark.unit
def test_samples_are_reproducible_across_runs():
    tools = [
        ToolSpec(
            name="tool",
            description="test tool",
            parameters=[
                ToolParameter(
                    name="x",
                    schema={"type": "integer", "minimum": 0, "maximum": 100},
                    required=True,
                    description="",
                )
            ],
        )
    ]

    mcp1 = MCPTune(model="test", mcpserver=None, seed=999)
    mcp2 = MCPTune(model="test", mcpserver=None, seed=999)

    d1 = mcp1.build_dataset(tools, samples_per_tool=10)
    d2 = mcp2.build_dataset(tools, samples_per_tool=10)

    assert d1 == d2


@pytest.mark.unit
def test_different_samples_are_generated_within_tool():
    tools = [
        ToolSpec(
            name="tool",
            description="test tool",
            parameters=[
                ToolParameter(
                    name="x",
                    schema={"type": "integer", "minimum": 0, "maximum": 10},
                    required=True,
                    description="",
                )
            ],
        )
    ]

    mcp = MCPTune(model="test", mcpserver=None, seed=42)

    dataset = mcp.build_dataset(tools, samples_per_tool=20)

    # extract generated values
    values = [row.arguments["x"] for row in dataset]

    # Not all identical (this would indicate broken RNG usage)
    assert len(set(values)) > 1


@pytest.mark.unit
def test_tool_grouping_preserved():
    tools = [
        ToolSpec(
            name="a",
            description="a",
            parameters=[
                ToolParameter(name="x", schema={"type": "integer"}, required=True, description="")
            ],
        ),
        ToolSpec(
            name="b",
            description="b",
            parameters=[
                ToolParameter(name="x", schema={"type": "integer"}, required=True, description="")
            ],
        ),
    ]

    mcp = MCPTune(model="test", mcpserver=None, seed=1)

    dataset = mcp.build_dataset(tools, samples_per_tool=3)

    counts = {"a": 0, "b": 0}
    for row in dataset:
        counts[row.tool_name] += 1

    assert counts["a"] == 3
    assert counts["b"] == 3
