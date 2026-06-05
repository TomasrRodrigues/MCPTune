import pytest

from mcptune.schema.dataset import DatasetRow
from mcptune.schema.tools import ToolSpec


@pytest.mark.integration
async def test_discover_returns_known_tools(tuner):
    tools = await tuner.adapter.discover_tools()

    assert {t.name for t in tools} == {"get_weather", "add"}


@pytest.mark.integration
async def test_discover_yields_toolspecs(tuner):
    tools = await tuner.adapter.discover_tools()
    assert all(isinstance(t, ToolSpec) for t in tools)


@pytest.mark.integration
async def test_build_dataset_produces_one_row_per_tool(tuner):
    tools = await tuner.adapter.discover_tools()
    dataset = tuner.build_dataset(tools)

    assert len(dataset) == len(tools)
    assert all(isinstance(row, DatasetRow) for row in dataset)


@pytest.mark.integration
async def test_dataset_rows_carry_well_formed_requests(tuner):
    tools = await tuner.adapter.discover_tools()
    dataset = tuner.build_dataset(tools)

    for row in dataset:
        assert row.request["jsonrpc"] == "2.0"
        assert row.request["params"]["name"] == row.tool_name
        assert row.request["params"]["arguments"] == row.arguments
