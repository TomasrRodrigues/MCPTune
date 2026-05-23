from dataclasses import asdict
import json
from pathlib import Path
from mcptune.schema.dataset import DatasetRow

SCHEMA_VERSION=1

def write_jsonl(rows:list[DatasetRow],path: str | Path) -> None:
    """
    write dataset rows to a JSONL file.
    """
    path=Path(path)

    with path.open("w",encoding="utf-8") as f:
        for row in rows:
            record=asdict(row)
            record["schema_version"]=SCHEMA_VERSION
            f.write(json.dumps(record))
            f.write("\n")


def read_jsonl(path:str | Path) -> list[DatasetRow]:
    """
    read dataset rows from a JSONL file.
    """

    path=Path(path)
    rows:list[DatasetRow]=[]
    with path.open("r",encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data=json.loads(line)
            data.pop("schema_version",None)

            row =DatasetRow(
                tool_name=data["tool_name"],
                arguments=data["arguments"],
                request=data["request"],
                response=data.get("response"),
                error=data.get("error"),
            )

            rows.append(row)
    return rows
