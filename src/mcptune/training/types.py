from dataclasses import dataclass


@dataclass
class TrainedModel:
    model_path: str | None = None
    backend: str | None = None
    metadata: dict | None = None
