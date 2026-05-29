import json
import os
from collections.abc import Callable
from importlib.resources import files
from typing import Any

from .cache.samplercache import SemanticSamplerCache
from .lookups import lookup_value


class SemanticSampler:
    """Generates semantically plausible argument values from schema metadata.

    Composes with the recursive sampler: this class returns a partial dict
    containing only the parameters it recognizes; the recursive sampler
    structurally samples whatever's missing.

    Backends:
        local         — offline lookup table, no model
        ollama        — local Ollama HTTP API (http://localhost:11434)
        transformers  — HuggingFace transformers, in-process
        none          — skip semantic sampling entirely
    """

    def __init__(
        self,
        backend: str = "local",
        prompt_version: str = "semantic_v1",
        model: str | None = None,
        temperature: float = 0.0,
        llm_call: Callable[[str], str] | None = None,
        ollama_host: str | None = None,
    ):
        self.backend = backend
        self.prompt_version = prompt_version
        self.model = model
        self.temperature = temperature
        self.ollama_host = ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self._llm_call = llm_call
        self.cache = SemanticSamplerCache()

        # Lazy-loaded transformers state — populated on first call.
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_model_name: str | None = None

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

            # If LLM returned nothing usable (empty parse, all values dropped by
            # type validation, or exception above), try local before giving up.
            if not generated:
                generated = self._execute_local_lookup(properties)

        if generated:
            self.cache.set(tool_name, tool_description, properties, generated)

        return generated

    # ---------------------------------------------------------------------
    # Local lookup (offline)
    # ---------------------------------------------------------------------

    def _execute_local_lookup(self, properties: dict[str, Any]) -> dict[str, Any]:
        results: dict[str, Any] = {}
        for name, schema in properties.items():
            value = lookup_value(name, schema)
            if value is not None:
                results[name] = value
        return results

    # ---------------------------------------------------------------------
    # LLM-backed batch generation
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
        """Dispatch to the configured backend. The llm_call override is the
        injection point for tests and custom callers."""
        if self._llm_call is not None:
            return self._llm_call(prompt)

        if self.backend == "ollama":
            return self._call_ollama(prompt)
        if self.backend == "transformers":
            return self._call_transformers(prompt)

        raise ValueError(f"No LLM caller configured for backend={self.backend!r}")

    def _call_ollama(self, prompt: str) -> str:
        """Call a local Ollama server. Requires `ollama serve` running and
        the requested model pulled. No API key, no signup, no cost."""
        import httpx  # transitive via fastmcp

        model = self.model or "qwen2.5:7b"
        try:
            response = httpx.post(
                f"{self.ollama_host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": self.temperature},
                    "format": "json",
                },
                timeout=120.0,
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Could not reach Ollama at {self.ollama_host}. "
                f"Is `ollama serve` running? Install: https://ollama.com"
            ) from e
        return response.json().get("response", "")

    def _call_transformers(self, prompt: str) -> str:
        """Call a HuggingFace transformers model in-process. Lazy-imports
        and lazy-loads the model so users without the extra installed can
        still use other backends."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Transformers backend requires the transformers and torch packages. "
                "Install with: pip install mcptune[transformers]"
            ) from e

        model_name = self.model or "Qwen/Qwen2.5-1.5B-Instruct"

        if self._hf_model is None or self._hf_model_name != model_name:
            self._hf_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._hf_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            self._hf_model_name = model_name

        messages = [{"role": "user", "content": prompt}]
        formatted = self._hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._hf_tokenizer(formatted, return_tensors="pt").to(self._hf_model.device)

        do_sample = self.temperature > 0
        with torch.no_grad():
            outputs = self._hf_model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=self.temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self._hf_tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        return self._hf_tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

    # ---------------------------------------------------------------------
    # Prompt building and response parsing
    # ---------------------------------------------------------------------

    def _build_prompt(
        self,
        tool_name: str,
        tool_description: str,
        properties: dict[str, Any],
    ) -> str:
        template = self._load_prompt_template()
        properties_json = json.dumps(properties, indent=2)
        return (
            template.replace("{tool_name}", tool_name)
            .replace("{tool_description}", tool_description or "(no description)")
            .replace("{properties_json}", properties_json)
        )

    def _load_prompt_template(self) -> str:
        return (
            files("mcptune.sampling")
            .joinpath("prompts", f"{self.prompt_version}.txt")
            .read_text(encoding="utf-8")
        )

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
