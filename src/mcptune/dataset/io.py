import json
from dataclasses import asdict
from pathlib import Path

from mcptune.dataset.validate import DatasetValidationError, validate_dataset_row_dict
from mcptune.schema.dataset import DatasetRow

SCHEMA_VERSION=1

def write_jsonl(rows:list[DatasetRow], path: str | Path) -> None:
    """
    Takes in memory dataset rows and writes them to a .jsonl file.
    
    Arguments:
        rows: list[DatasetRow]
        path: str
    
    Returnable:
        void (file written)
    """
    path=Path(path)

    with path.open("w",encoding="utf-8") as f:
        for row in rows:
            record = asdict(row)
            record["schema_version"] = SCHEMA_VERSION
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path, strict: bool = True) -> list[DatasetRow]:
    """
    Reads the jsonl files and constructs DatasetRow objects.
    
    Arguments: 
        path: str
    
    Returnable:
        dataset: list[DatasetRow]
    """

    path=Path(path)
    rows:list[DatasetRow]=[]

    with path.open("r",encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            
            data = json.loads(line)
            #schema_version = data.pop("schema_version", None)

            try:
                validate_dataset_row_dict(data)
            except DatasetValidationError:
                if strict:
                    raise
                continue
            

            row = DatasetRow(
                tool_name=data["tool_name"],
                arguments=data["arguments"],
                request=data["request"],
                response=data.get("response"),
                error=data.get("error"),
            )

            rows.append(row)
    return rows
