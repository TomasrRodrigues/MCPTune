from .base import ArgumentSampler

import re
import random
import string


class PrimitiveSampler(ArgumentSampler):

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def sample(self, schema: dict):

        t = schema.get("type")

        if t == "string":
            return self._string(schema)

        if t == "integer":
            return self._integer(schema)

        if t == "number":
            return self._number(schema)

        if t == "boolean":
            return self.rng.choice([True, False])

        return None

    def _string(self, schema: dict):
        fmt = schema.get("format")
        pattern = schema.get("pattern")

        if fmt == "email":
            return self._email()

        min_len = schema.get("minLength", 3)
        max_len = schema.get("maxLength", 12)
        length = self.rng.randint(min_len, max_len)

        if pattern:
            try:
                import rstr
                return rstr.xeger(pattern)
            except Exception:
                return "".join(self.rng.choices(string.ascii_lowercase, k=length))

        return "".join(self.rng.choices(string.ascii_lowercase, k=length))

    def _integer(self, schema: dict):
        min_v = schema.get("minimum", 0)
        max_v = schema.get("maximum", min_v + 100)

        if min_v > max_v:
            max_v = min_v + 100

        return self.rng.randint(min_v, max_v)

    def _number(self, schema: dict):
        min_v = schema.get("minimum", 0.0)
        max_v = schema.get("maximum", min_v + 100.0)

        if min_v > max_v:
            max_v = min_v + 100.0

        return self.rng.uniform(min_v, max_v)

    def _email(self):
        user = "".join(self.rng.choices(string.ascii_lowercase, k=6))
        domain = "".join(self.rng.choices(string.ascii_lowercase, k=5))
        return f"{user}@{domain}.com"