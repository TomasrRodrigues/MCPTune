import random
from typing import Any

from .base import ArgumentSampler
from .primitive import PrimitiveSampler


class RecursiveSampler(ArgumentSampler):
    MAX_DEPTH = 10

    def __init__(self, rng=None, semantic_sampler=None):
        self.rng = rng or random.Random()
        self.primitive = PrimitiveSampler(self.rng)
        self.semantic_sampler = semantic_sampler

    def sample(
        self,
        schema: dict,
        depth: int = 0,
        tool_name: str = "",
        tool_description: str = "",
    ) -> Any:
        if depth >= self.MAX_DEPTH:
            return self._fallback(schema)

        if "default" in schema:
            return schema["default"]

        if "enum" in schema:
            return self.rng.choice(schema["enum"])

        if "anyOf" in schema:
            return self.sample(self.rng.choice(schema["anyOf"]), depth + 1)

        if "oneOf" in schema:
            return self.sample(self.rng.choice(schema["oneOf"]), depth + 1)

        if self._is_nullable(schema):
            return self._sample_nullable(schema, depth)

        schema_type = schema.get("type")

        if schema_type == "object":
            return self._sample_object(schema, depth, tool_name, tool_description)

        if schema_type == "array":
            return self._sample_array(schema, depth)

        return self.primitive.sample(schema)

    def _sample_object(
        self,
        schema: dict,
        depth: int,
        tool_name: str,
        tool_description: str,
    ) -> dict[str, Any]:
        """Per-parameter composition: semantic sampler fills what it knows,
        structural sampling fills the rest."""
        properties = schema.get("properties", {})
        required = set(schema.get("required", []))

        semantic_values: dict[str, Any] = {}
        if self.semantic_sampler and self.semantic_sampler.backend != "none":
            try:
                semantic_values = self.semantic_sampler.sample_batch(
                    tool_name=tool_name,
                    tool_description=tool_description,
                    properties=properties,
                )
            except Exception as e:
                print(f"[MCPTune Warn] Semantic sampler skipped: {e}")

        result: dict[str, Any] = {}
        for name, subschema in properties.items():
            if name in semantic_values:
                result[name] = semantic_values[name]
            elif name in required:
                result[name] = self.sample(subschema, depth + 1)
            elif subschema.get("type") == "object":
                result[name] = self.sample(subschema, depth + 1)
            elif self.rng.random() < 0.7:
                result[name] = self.sample(subschema, depth + 1)

        return result

    def _is_nullable(self, schema: dict) -> bool:
        return schema.get("nullable") is True or (
            isinstance(schema.get("type"), list) and "null" in schema["type"]
        )

    def _sample_nullable(self, schema: dict, depth: int) -> Any:
        if self.rng.random() < 0.5:
            return None
        new_schema = {k: v for k, v in schema.items() if k != "nullable"}
        t = new_schema.get("type")
        if isinstance(t, list):
            new_schema["type"] = [x for x in t if x != "null"]
        return self.sample(new_schema, depth + 1)

    def _sample_array(self, schema: dict, depth: int) -> list:
        item_schema = schema.get("items", {})
        min_items = schema.get("minItems", 1)
        max_items = schema.get("maxItems", 5)
        length = self.rng.randint(min_items, max_items)
        return [self.sample(item_schema, depth + 1) for _ in range(length)]

    def _fallback(self, schema: dict) -> Any:
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
