import random
import string

from .base import ArgumentSampler


class PrimitiveSampler(ArgumentSampler):
    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def sample(self, schema: dict):

        t = schema.get("type")

        if t == "string":
            return self._string(schema)

        if t == "integer":
            return self.rng.randint(0, 100)

        if t == "number":
            return self.rng.uniform(0, 100)

        if t == "boolean":
            return self.rng.choice([True, False])

        return None

    def _string(self, schema):
        return "".join(self.rng.choices(string.ascii_lowercase, k=8))
