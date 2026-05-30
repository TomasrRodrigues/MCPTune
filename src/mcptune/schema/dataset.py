from dataclasses import dataclass


@dataclass
class DatasetRow:
    tool_name: str
    arguments: dict
    request: dict
    response: dict | None = None
    error: str | None = None
    user_intent: str | None = None
    intent_prompt_version: str | None = None
