import hashlib
import json
from typing import Any


class SemanticSamplerCache:
    """Class to define the cache to store the arguments and tools for the LLM to generate data more efficiently"""

    def __init__(self):
        """"""
        self._storage: dict[str, dict[str, Any]] = {}

    def _generate_key(
        self, tool_name: str, tool_description: str, properties_schema: dict[str, Any]
    ) -> str:
        """
        Creates a deterministic SHA256 fingerprint from the tool metadata.
        Sorting keys ensures {"a": 1, "b": 2} and {"b": 2, "a": 1} produce identical hashes.
        """
        normalized_payload = {
            "name": tool_name,
            "description": tool_description,
            "properties": properties_schema,
        }

        serialized_bytes = json.dumps(normalized_payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(serialized_bytes).hexdigest()

    def get(
        self, tool_name: str, tool_description: str, properties_schema: dict[str, Any]
    ) -> dict[str, Any] | None:
        """Retrieves cached batch arguments for a tool if they exist."""
        key = self._generate_key(tool_name, tool_description, properties_schema)
        return self._storage.get(key)

    def set(
        self,
        tool_name: str,
        tool_description: str,
        properties_schema: dict[str, Any],
        values: dict[str, Any],
    ) -> None:
        """Stores the full batch of generated arguments for a tool."""
        key = self._generate_key(tool_name, tool_description, properties_schema)
        self._storage[key] = values

    def clear(self) -> None:
        """Clears the cache (useful between test runs)."""
        self._storage.clear()
