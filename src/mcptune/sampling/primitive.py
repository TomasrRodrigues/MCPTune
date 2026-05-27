from .base import ArgumentSampler

import re
import random
import string


class PrimitiveSampler(ArgumentSampler):

    def sample(self, schema: dict):

        t = schema.get("type")

        if t == "string":
            return self._string(schema)

        if t == "integer":
            return self._integer(schema)

        if t == "number":
            return self._number(schema)

        if t == "boolean":
            return random.choice([True, False])

        return None



    def _string(self, schema: dict):
        fmt = schema.get("format")
        pattern = schema.get("pattern")

        if fmt == "email":
            return self._email()

        min_len = schema.get("minLength", 3)
        max_len = schema.get("maxLength", 12)
        length = random.randint(min_len, max_len)

        if pattern:
            # best-effort: do NOT crash, just fallback safe string
            try:
                # optional dependency approach (not required by issue)
                import rstr
                return rstr.xeger(pattern)
            except Exception:
                # safe fallback
                return "".join(random.choices(string.ascii_lowercase, k=length))

        return "".join(random.choices(string.ascii_lowercase, k=length))



    def _integer(self, schema: dict):
        min_v = schema.get("minimum", 0)
        max_v = schema.get("maximum", min_v + 100)

        if min_v > max_v:
            max_v = min_v + 100

        return random.randint(min_v, max_v)



    def _number(self, schema: dict):
        min_v = schema.get("minimum", 0.0)
        max_v = schema.get("maximum", min_v + 100.0)

        if min_v > max_v:
            max_v = min_v + 100.0

        return random.uniform(min_v, max_v)
    
    def _email(self):
        user = "".join(random.choices(string.ascii_lowercase, k=6))
        domain = "".join(random.choices(string.ascii_lowercase, k=5))
        return f"{user}@{domain}.com"