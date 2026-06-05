from abc import ABC, abstractmethod
from typing import Any

from mcptune.schema.dataset import DatasetRow
from mcptune.schema.tools import ToolSpec


class Format(ABC):
    needs_tools: bool = False

    @abstractmethod
    def format_tool(
        self, rows: list[DatasetRow], tools: list[ToolSpec] | None = None
    ) -> list[dict[str, Any]]:
        raise NotImplementedError()
