"""Schema dataclasses for on-disk dataset rows.

Contains the `DatasetRow` dataclass used across the codebase to
represent a single tool-use example including the user intent, the
tool call request, and (optionally) the execution response.
"""

from dataclasses import dataclass


@dataclass
class DatasetRow:
    """Represents a single dataset example.

    Fields
    ------
    tool_name: str
        The tool identifier.
    arguments: dict
        Arguments passed to the tool.
    request: dict
        The MCP request object used to call the tool.
    response: dict | None
        Optional execution response.
    error: str | None
        Optional error string.
    user_intent: str | None
        Natural-language intent describing the tool usage.
    intent_prompt_version: str | None
        Version identifier for the prompt used to generate `user_intent`.
    """

    tool_name: str
    arguments: dict
    request: dict
    response: dict | None = None
    error: str | None = None
    user_intent: str | None = None
    intent_prompt_version: str | None = None
    final_answer: str | None = None
    answer_prompt_version: str | None = None
