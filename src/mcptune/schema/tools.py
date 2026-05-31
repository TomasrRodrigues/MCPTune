"""Tool metadata schema used by MCPTune.

Defines small dataclasses representing tool parameter definitions and
the normalized `ToolSpec` used throughout the pipeline.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolParameter:
    """Definition of a single tool parameter."""

    name: str
    schema: dict
    required: bool
    description: str


@dataclass
class ToolSpec:
    """Normalized tool specification.

    `parameters` is a list of `ToolParameter` instances describing the
    input schema. `raw_input_schema` retains the original schema provided
    by the MCP server for convenience.
    """

    name: str
    description: str
    parameters: list[ToolParameter]
    outputSchema: ToolParameter | None = None
    raw_input_schema: dict[str, Any] = field(default_factory=dict)
