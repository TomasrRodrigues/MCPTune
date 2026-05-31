"""Intent example datatypes.

Lightweight dataclasses used by intent-related utilities and tests.
"""

from dataclasses import dataclass
from typing import Any


@dataclass
class IntentExample:
    """Represents a single synthetic intent example."""

    user_message: str
    tool_name: str
    arguments: dict[str, Any]
