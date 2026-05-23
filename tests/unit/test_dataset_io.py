from mcptune.dataset.io import write_jsonl, read_jsonl
from mcptune.schema.dataset import DatasetRow


def test_jsonl_round_trip(tmp_path):
    rows = [
        DatasetRow(
            tool_name="weather_tool",
            arguments={"city": "Delhi"},
            request={"prompt": "Weather in Delhi"},
            response={"temp": 35},
            error=None,
        )
    ]

    output_file = tmp_path / "dataset.jsonl"

    write_jsonl(rows, output_file)

    loaded_rows = read_jsonl(output_file)

    assert loaded_rows == rows