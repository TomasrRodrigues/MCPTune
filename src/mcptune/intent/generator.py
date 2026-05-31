"""Tiny intent generator for quick, dependency-free examples.

This module provides a low-effort heuristic that turns a tool name
into a short natural-language request. It is primarily useful for
tests and examples where a full LLM-backed intent synthesizer is not
available.
"""

import random

from mcptune.schema import ToolSpec


class IntentGenerator:
    """Heuristic intent generator.

    The generator uses a seeded RNG so results are deterministic when a
    `seed` is provided.
    """

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)

    def generate(self, tool: ToolSpec) -> str:
        name = tool.name.replace("_", " ")

        templates = [
            f"give me {name}",
            f"I need {name}",
            f"can you show me {name}",
            f"what is the {name}",
        ]

        return self.rng.choice(templates)
