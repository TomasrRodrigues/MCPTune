"""No-op trainer backend for fast unit tests and dry runs.

Mirrors the TrainerBackend interface but performs no actual training -
records metadata about what would be trained and returns a TrainedModel
descriptor. Useful for:

- Unit testing pipeline orchestration without heavy dependencies
- Validating dataset shape before kicking off a real training run
- CI builds where transformers/torch aren't installed
"""

from __future__ import annotations

from pathlib import Path

from ..base import TrainerBackend
from ..types import TrainedModel


class MockTrainerBackend(TrainerBackend):
    """Records training metadata without performing any training."""

    def train(
        self,
        model_name: str,
        dataset,
        config: dict | None = None,
        tools=None,
    ) -> TrainedModel:
        return TrainedModel(
            backend="mock",
            metadata={
                "model_name": model_name,
                "num_examples": len(dataset),
                "config": config or {},
                "num_tools": len(tools) if tools else 0,
            },
        )

    def save(self, model: TrainedModel, path: str) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        model.model_path = path
