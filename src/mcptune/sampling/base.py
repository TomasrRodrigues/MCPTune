"""Sampler base interfaces.

Defines the minimal `ArgumentSampler` abstract class used by
sampling implementations to produce values that satisfy JSON Schema
fragments used in tool parameter definitions.
"""

from abc import ABC, abstractmethod
from typing import Any


class ArgumentSampler(ABC):
    """Abstract interface for argument samplers."""

    @abstractmethod
    def sample(self, schema: dict[str, Any]) -> Any:
        """Generate a valid value for a given JSON Schema fragment.

        Implementations must return a JSON-serializable value that
        matches the provided schema where possible.
        """
        raise NotImplementedError()
