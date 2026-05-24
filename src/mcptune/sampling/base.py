from abc import ABC, abstractmethod
from typing import Any


class ArgumentSampler(ABC):
    @abstractmethod
    def sample(self, schema: dict[str, Any]) -> Any:
        """
        Generate a valid value for a given JSONSchema fragment.
        """
        pass
