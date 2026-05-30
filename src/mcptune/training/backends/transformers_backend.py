from ..base import TrainerBackend
from ..types import TrainedModel


class TransformersTrainerBackend(TrainerBackend):
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
