from abc import ABC, abstractmethod

from .types import TrainedModel


class TrainerBackend(ABC):
    @abstractmethod
    def train(
        self,
        model_name: str,
        dataset,
        config: dict | None = None,
    ) -> TrainedModel:
        pass

    @abstractmethod
    def save(
        self,
        model: TrainedModel,
        path: str,
    ) -> None:
        pass
