"""Simple descriptor for trained model artifacts."""

from dataclasses import dataclass
from typing import Any


@dataclass
class TrainedModel:
    """Metadata describing training outputs.

    Fields
    ------
    model_path: str | None
        Path where model artifacts are stored (if any).
    backend: str | None
        Identifier for the training backend used.
    metadata: dict | None
        Backend-specific metadata (e.g., hyperparameters, example counts).
    """

    model_path: str | None = None
    backend: str | None = None
    metadata: dict[str, Any] | None = None
