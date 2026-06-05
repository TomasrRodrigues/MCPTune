"""Lightweight local intent backend.

This backend provides a deterministic, dependency-free way to
produce placeholder intent descriptions for tools. It is useful for
fast development, testing and when no LLM-backed intent generation
is available.
"""

from typing import Any

from mcptune.schema.tools import ToolSpec


class LocalIntentBackend:
    """Simple heuristic intent generator.

    Produces a short intent string and a lightweight `arguments_hint`
    mapping that mirrors the parameter names. The output shape is
    intentionally minimal and meant to be compatible with the
    higher-level `IntentSynthesizer` interface.
    """

    def generate_intent(self, tool: ToolSpec) -> dict[str, Any]:
        """Return a placeholder intent for `tool`.

        Parameters
        ----------
        tool:
            Tool specification for which to generate a minimal intent.

        Returns
        -------
        dict
            A dict containing `intent` (string) and `arguments_hint`
            mapping parameter names to simple placeholders.
        """
        return {
            "intent": f"use {tool.name}",
            "arguments_hint": {
                p.name: p.name  # placeholder like "city", "temperature"
                for p in tool.parameters
            },
        }
