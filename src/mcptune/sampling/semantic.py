from typing import Any

from .cache.samplercache import SemanticSamplerCache


class SemanticSampler:
    """Generates semantically plausible argument values from schema metadata.

    Composes with the recursive sampler: this class returns a partial dict
    containing only the parameters it recognizes; the recursive sampler
    structurally samples whatever's missing.
    """

    def __init__(self, backend: str = "local", prompt_version: str = "semantic_v1"):
        self.backend = backend
        self.prompt_version = prompt_version
        self.cache = SemanticSamplerCache()

    def sample_batch(
        self,
        tool_name: str,
        tool_description: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Generate values for as many of `properties` as this sampler can handle.
        Returns a (possibly partial) dict; missing keys are intentional."""
        if not properties or self.backend == "none":
            return {}

        cached = self.cache.get(tool_name, tool_description, properties)
        if cached is not None:
            return cached

        if self.backend == "local":
            generated = self._execute_local_lookup(properties)
        elif self.backend in ("anthropic", "openai"):
            try:
                generated = self._execute_llm_batch(tool_name, tool_description, properties)
            except Exception:
                generated = self._execute_local_lookup(properties)
        else:
            generated = {}

        if generated:
            self.cache.set(tool_name, tool_description, properties, generated)

        return generated

    def _execute_local_lookup(self, properties: dict[str, Any]) -> dict[str, Any]:
        """Offline / CI fallback. Returns only recognized parameters; unknown
        parameters are silently omitted so the structural sampler handles them."""
        results: dict[str, Any] = {}
        lookup = {
            "city": "Lisbon",
            "email": "test@example.com",
            "url": "https://google.com",
        }

        for name, schema in properties.items():
            name_lower = name.lower()
            fmt = schema.get("format", "")

            if "email" in fmt or "email" in name_lower:
                results[name] = lookup["email"]
            elif "city" in name_lower or "location" in name_lower:
                results[name] = lookup["city"]
            elif "url" in fmt or "uri" in name_lower or "url" in name_lower:
                results[name] = lookup["url"]
            # else: omit; the recursive sampler fills it in structurally

        return results

    def _execute_llm_batch(
        self,
        tool_name: str,
        tool_description: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        """Placeholder until PR C wires up real Anthropic / OpenAI clients."""
        raise NotImplementedError("LLM backend not yet implemented; use backend='local' or 'none'")
