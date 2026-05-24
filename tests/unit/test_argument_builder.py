import pytest

from mcptune import MCPTune
from mcptune.schema.tools import ToolParameter, ToolSpec


class DummySampler:
    """Deterministic sampler returning a sentinel value, so tests can
    assert structure without depending on random output."""

    def sample(self, schema):
        return "x"


@pytest.mark.unit
def test_build_arguments_maps_each_parameter():
    tool = ToolSpec(
        name="test",
        description="",
        parameters=[
            ToolParameter(name="city", schema={"type": "string"}, required=True, description=""),
            ToolParameter(
                name="country", schema={"type": "string"}, required=False, description=""
            ),
        ],
    )

    m = MCPTune(model="x", mcpserver=None)
    m.sampler = DummySampler()

    args = m.build_arguments(tool)

    assert set(args.keys()) == {"city", "country"}
    assert all(v == "x" for v in args.values())


@pytest.mark.unit
def test_build_arguments_no_parameters_returns_empty():
    tool = ToolSpec(name="test", description="", parameters=[])
    m = MCPTune(model="x", mcpserver=None)

    assert m.build_arguments(tool) == {}


@pytest.mark.unit
def test_seeded_dataset_build_is_reproducible_across_instances():
    tools = [
        ToolSpec(
            name="weather",
            description="",
            parameters=[
                ToolParameter(
                    name="city",
                    schema={"type": "string"},
                    required=True,
                    description="",
                ),
                ToolParameter(
                    name="limit",
                    schema={"type": "integer"},
                    required=False,
                    description="",
                ),
            ],
        )
    ]
    first = MCPTune(model="x", mcpserver=None, seed=42)
    second = MCPTune(model="x", mcpserver=None, seed=42)

    assert first.build_dataset(tools) == second.build_dataset(tools)
