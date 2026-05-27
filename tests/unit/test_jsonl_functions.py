import os
import tempfile

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
