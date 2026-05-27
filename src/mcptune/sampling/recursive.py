from .base import ArgumentSampler
from .primitive import PrimitiveSampler
import random


class RecursiveSampler(ArgumentSampler):
    MAX_DEPTH = 10

    def __init__(self, rng=None):
        self.rng = rng or random.Random()
        self.primitive = PrimitiveSampler(self.rng)

    def sample(self, schema: dict, depth: int = 0):

        if depth >= self.MAX_DEPTH:
            return self._fallback(schema)

        if "default" in schema:
            return schema["default"]

        if "enum" in schema:
            return self._sample_enum(schema)

        if "anyOf" in schema:
            return self._sample_anyof(schema, depth)

        if "oneOf" in schema:
            return self._sample_oneof(schema, depth)

        if self._is_nullable(schema):
            return self._sample_nullable(schema, depth)

        schema_type = schema.get("type")

        if schema_type == "object":
            return self._sample_object(schema, depth)

        if schema_type == "array":
            return self._sample_array(schema, depth)

        return self.primitive.sample(schema)

    def _sample_enum(self, schema: dict):
        return self.rng.choice(schema["enum"])

    def _sample_anyof(self, schema: dict, depth: int):
        branch = self.rng.choice(schema["anyOf"])
        return self.sample(branch, depth + 1)

    def _sample_oneof(self, schema: dict, depth: int):
        branch = self.rng.choice(schema["oneOf"])
        return self.sample(branch, depth + 1)

    def _is_nullable(self, schema: dict):
        return (
            schema.get("nullable") is True
            or (
                isinstance(schema.get("type"), list)
                and "null" in schema["type"]
            )
        )

    def _sample_nullable(self, schema: dict, depth: int):
        if self.rng.random() < 0.5:
            return None

        new_schema = {k: v for k, v in schema.items() if k != "nullable"}

        t = new_schema.get("type")
        if isinstance(t, list):
            new_schema["type"] = [x for x in t if x != "null"]

        return self.sample(new_schema, depth + 1)

    def _sample_object(self, schema: dict, depth: int):
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        result = {}

        # required fields always included
        for name in required:
            if name in properties:
                result[name] = self.sample(properties[name], depth + 1)

        # optional fields
        for name, subschema in properties.items():
            if name in required:
                continue

            # FIX: ensure nested object stability
            if subschema.get("type") == "object":
                result[name] = self.sample(subschema, depth + 1)
            elif self.rng.random() < 0.7:
                result[name] = self.sample(subschema, depth + 1)

        return result

    def _sample_array(self, schema: dict, depth: int):
        item_schema = schema.get("items", {})

        min_items = schema.get("minItems", 1)
        max_items = schema.get("maxItems", 5)

        length = self.rng.randint(min_items, max_items)

        return [
            self.sample(item_schema, depth + 1)
            for _ in range(length)
        ]

    def _fallback(self, schema: dict):
        t = schema.get("type")

        if t == "object":
            return {}

        if t == "array":
            return []

        if t == "string":
            return "x"

        if t == "integer":
            return 0

        if t == "number":
            return 0.0

        if t == "boolean":
            return False

        return None
