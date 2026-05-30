import os
import tempfile

import pytest

from mcptune.dataset.io import read_jsonl, write_jsonl
from mcptune.schema.dataset import DatasetRow


def test_jsonl_round_trip():
    rows = [
        DatasetRow(
            tool_name="get_weather",
            arguments={"city": "Porto"},
            request={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "tools/call",
                "params": {
                    "name": "get_weather",
                    "arguments": {"city": "Porto"},
                },
            },
        ),
        DatasetRow(
            tool_name="add",
            arguments={"a": 1, "b": 2},
            request={
                "jsonrpc": "2.0",
                "id": "2",
                "method": "tools/call",
                "params": {
                    "name": "add",
                    "arguments": {"a": 1, "b": 2},
                },
            },
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "dataset.jsonl")

        # write
        write_jsonl(rows, path)

        # read
        loaded = read_jsonl(path)

        # compare lengths
        assert len(rows) == len(loaded)

        for original, restored in zip(rows, loaded, strict=True):
            assert original.tool_name == restored.tool_name
            assert original.arguments == restored.arguments
            assert original.request == restored.request
            assert original.response == restored.response
            assert original.error == restored.error


@pytest.mark.unit
def test_jsonl_round_trip_preserves_user_intent_and_prompt_version(tmp_path):
    rows = [
        DatasetRow(
            tool_name="get_weather",
            arguments={"city": "Lisbon"},
            request={
                "jsonrpc": "2.0",
                "id": "1",
                "method": "tools/call",
                "params": {"name": "get_weather", "arguments": {"city": "Lisbon"}},
            },
            user_intent="What's the weather in Lisbon?",
            intent_prompt_version="intent_v1",
        ),
        DatasetRow(
            tool_name="ping",
            arguments={},
            request={
                "jsonrpc": "2.0",
                "id": "2",
                "method": "tools/call",
                "params": {"name": "ping", "arguments": {}},
            },
            # template fallback case - no LLM was used
            user_intent="Use the ping tool.",
            intent_prompt_version=None,
        ),
    ]

    path = tmp_path / "dataset.jsonl"
    write_jsonl(rows, path)
    loaded = read_jsonl(path)

    assert len(loaded) == 2
    assert loaded[0].user_intent == "What's the weather in Lisbon?"
    assert loaded[0].intent_prompt_version == "intent_v1"
    assert loaded[1].user_intent == "Use the ping tool."
    assert loaded[1].intent_prompt_version is None
