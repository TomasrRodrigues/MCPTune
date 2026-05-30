import json
from collections.abc import Callable
from importlib.resources import files
from typing import Any

from mcptune.llm.client import LLMClient

from .cache.samplercache import SemanticSamplerCache
from .lookups import lookup_value


class SemanticSampler:
    """Generates semantically plausible argument values from schema metadata.

    Composes with the recursive sampler: this class returns a partial dict
    containing only the parameters it recognizes; the recursive sampler
    structurally samples whatever's missing.
    """

    def __init__(
        self,
        backend: str = "local",
        prompt_version: str = "grounded_semantic_v1",
        model: str | None = None,
        temperature: float = 0.0,
        llm_call: Callable[[str], str] | None = None,
        ollama_host: str | None = None,
    ):
        self.backend = backend
        self.prompt_version = prompt_version
        self.model = model
        self.temperature = temperature
        self.ollama_host = ollama_host
        self._llm_call = llm_call
        self._client: LLMClient | None = None  # lazy
        self.cache = SemanticSamplerCache()

    def sample_batch(
        self,
        tool_name: str,
        tool_description: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        if not properties or self.backend == "none":
            return {}

        cached = self.cache.get(tool_name, tool_description, properties)
        if cached is not None:
            return cached

        generated: dict[str, Any] = {}

        if self.backend == "local":
            generated = self._execute_local_lookup(properties)
        elif self.backend in ("ollama", "transformers"):
            try:
                generated = self._execute_llm_batch(tool_name, tool_description, properties)
            except Exception as e:
                print(f"[MCPTune Warn] LLM backend failed, falling back to local: {e}")

            if not generated:
                generated = self._execute_local_lookup(properties)

        if generated:
            self.cache.set(tool_name, tool_description, properties, generated)

        return generated

    # ---------------------------------------------------------------------
    # Local lookup
    # ---------------------------------------------------------------------

    def _execute_local_lookup(self, properties: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for name, schema in properties.items():
            value = lookup_value(name, schema)
            if value is not None:
                results[name] = value
        return results

    # ---------------------------------------------------------------------
    # LLM-backed generation (via shared LLMClient)
    # ---------------------------------------------------------------------

    def _execute_llm_batch(
        self,
        tool_name: str,
        tool_description: str,
        properties: dict[str, Any],
    ) -> dict[str, Any]:
        prompt = self._build_prompt(tool_name, tool_description, properties)
        raw_response = self._call_llm(prompt)
        parsed = self._extract_json(raw_response)
        return self._validate_values(parsed, properties)

    def _call_llm(self, prompt: str) -> str:
        if self._llm_call is not None:
            return self._llm_call(prompt)
        if self._client is None:
            self._client = LLMClient(
                backend=self.backend,
                model=self.model,
                temperature=self.temperature,
                ollama_host=self.ollama_host,
            )
        return self._client.generate(prompt, json_mode=True)

    # ---------------------------------------------------------------------
    # Prompt building (unchanged from Issue 13)
    # ---------------------------------------------------------------------

    def _build_prompt(
        self,
        tool_name: str,
        tool_description: str,
        properties: dict[str, Any],
    ) -> str:
        template = self._load_prompt_template()
        parameter_block = self._format_parameter_block(properties)
        properties_json = json.dumps(properties, indent=2)
        tool_desc = self._truncate(tool_description or "(no description)", 800)

        return (
            template.replace("{tool_name}", tool_name)
            .replace("{tool_description}", tool_desc)
            .replace("{parameter_block}", parameter_block)
            .replace("{properties_json}", properties_json)
        )

    def _load_prompt_template(self) -> str:
        return (
            files("mcptune.sampling")
            .joinpath("prompts", f"{self.prompt_version}.txt")
            .read_text(encoding="utf-8")
        )

    @staticmethod
    def _format_parameter_block(properties: dict[str, Any]) -> str:
        lines = []
        for name, schema in properties.items():
            type_str = schema.get("type", "any")
            format_str = schema.get("format", "")
            description = (schema.get("description") or "").strip()

            type_part = type_str
            if format_str:
                type_part = f"{type_str}, format: {format_str}"

            header = f"- {name} ({type_part})"
            if description:
                description = SemanticSampler._truncate(description, max_chars=800)
                lines.append(f"{header}: {description}")
            else:
                lines.append(header)

        return "\n".join(lines)

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

    @staticmethod
    def _extract_json(response: str) -> dict:
        response = response.strip()
        try:
            parsed = json.loads(response)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass

        start = response.find("{")
        end = response.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}

        try:
            parsed = json.loads(response[start : end + 1])
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}

    @staticmethod
    def _validate_values(values: dict, properties: dict[str, Any]) -> dict[str, Any]:
        type_check = {
            "string": lambda v: isinstance(v, str),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
        }

        validated = {}
        for name, value in values.items():
            schema = properties.get(name)
            if schema is None:
                continue
            expected = schema.get("type")
            check = type_check.get(expected)
            if check is None or check(value):
                validated[name] = value

        return validated
