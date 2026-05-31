"""Lightweight Transformers training backend placeholder.

This backend is a stub used by tests and examples. It does not run
actual training; instead it returns a `TrainedModel` descriptor and
tracks the save path when requested.
"""

from ..base import TrainerBackend
from ..types import TrainedModel


class TransformersTrainerBackend(TrainerBackend):
    """Placeholder trainer that records metadata without training."""

    def train(
        self,
        model_name: str,
        dataset,
        config: dict | None = None,
    ) -> TrainedModel:
        return TrainedModel(
            backend="transformers",
            metadata={
                "model_name": model_name,
                "num_examples": len(dataset),
            },
        )

    def save(
        self,
        model: TrainedModel,
        path: str,
    ) -> None:
        model.model_path = path
