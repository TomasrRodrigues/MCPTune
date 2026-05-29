import random

from mcptune.schema import ToolSpec


class IntentGenerator:
    """
    Generates natural language user queries from tool schemas.
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
