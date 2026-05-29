from dataclasses import dataclass
from typing import Any


@dataclass
class IntentExample:
    user_message: str
    tool_name: str
    arguments: dict[str, Any]
