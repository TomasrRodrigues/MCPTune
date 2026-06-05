"""Training backend interface.

Defines the abstract `TrainerBackend` that concrete training backends
must implement. The interface is intentionally small: implementors
should provide `train` to consume a dataset and return a `TrainedModel`,
and `save` to persist artifacts.
"""

from abc import ABC, abstractmethod
from typing import Any

from ..schema.tools import ToolSpec
from .types import TrainedModel


class TrainerBackend(ABC):
    """Abstract trainer backend interface."""

    @abstractmethod
    def train(
        self,
        model_name: str,
        dataset: list[Any],
        config: dict[str, Any] | None = None,
        tools: list[ToolSpec] | None = None,
    ) -> TrainedModel:
        """Train on `dataset` and return a `TrainedModel` descriptor.

        `tools` is the available ToolSpec set, required by formats that
        render tool definitions into context (e.g. "tool_use").
        """
        raise NotImplementedError()

    @abstractmethod
    def save(
        self,
        model: TrainedModel,
        path: str,
    ) -> None:
        """Persist `model` to `path` (backend-specific)."""
        raise NotImplementedError()
