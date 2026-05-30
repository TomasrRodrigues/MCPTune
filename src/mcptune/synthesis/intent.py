"""Synthesize realistic user-facing prompts that would elicit each tool call."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from mcptune.llm.client import LLMClient
from mcptune.schema.tools import ToolSpec


@dataclass(frozen=True)
class IntentResult:
    """Result of an intent synthesis call.

    `prompt_version` is None when the template fallback was used — either
    because the backend is "none" or because the LLM call failed/returned
    empty. Callers recording provenance should treat None as "no LLM
    prompt was actually used to produce this intent."
    """

    intent: str
    prompt_version: str | None


class IntentSynthesizer:
    """Synthesizes a natural-language user message that would lead an
    assistant to call `tool` with `arguments`.

    Backends:
        none          — deterministic template fallback, no LLM (default)
        ollama        — local Ollama HTTP API
        transformers  — HuggingFace transformers, in-process
    """

    def __init__(
        self,
        backend: str = "none",
        prompt_version: str = "intent_v1",
        model: str | None = None,
        temperature: float = 0.7,
        llm_call: Callable[[str], str] | None = None,
        ollama_host: str | None = None,
    ):
        self.backend = backend
        self.prompt_version = prompt_version
        self.model = model
        self.temperature = temperature
        self.ollama_host = ollama_host
        self._llm_call = llm_call
        self._client: LLMClient | None = None

    def synthesize(self, tool: ToolSpec, arguments: dict[str, Any]) -> IntentResult:
        """Return an IntentResult. The `prompt_version` is None when the
        template fallback was used (backend='none', LLM error, or empty
        response). Otherwise it's the prompt version that produced the
        intent."""
        if self.backend == "none":
            return IntentResult(self._template_fallback(tool, arguments), None)

        try:
            prompt = self._build_prompt(tool, arguments)
            response = self._call_llm(prompt)
        except Exception as e:
            print(f"[MCPTune Warn] Intent synthesis failed, using template: {e}")
            return IntentResult(self._template_fallback(tool, arguments), None)

        text = response.strip()
        if not text:
            return IntentResult(self._template_fallback(tool, arguments), None)
        return IntentResult(text, self.prompt_version)

    # ---------------------------------------------------------------------

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
        return self._client.generate(prompt, json_mode=False)

    def _build_prompt(self, tool: ToolSpec, arguments: dict[str, Any]) -> str:
        template = self._load_prompt_template()

        param_lines = []
        for p in tool.parameters:
            desc = (p.description or "").strip()
            line = f"- {p.name}"
            if desc:
                line += f": {self._truncate(desc, 400)}"
            param_lines.append(line)
        parameters_section = "\n".join(param_lines) if param_lines else "(no parameters)"

        return (
            template.replace("{tool_name}", tool.name)
            .replace(
                "{tool_description}", self._truncate(tool.description or "(no description)", 800)
            )
            .replace("{parameters_section}", parameters_section)
            .replace("{arguments_json}", json.dumps(arguments, indent=2))
        )

    def _load_prompt_template(self) -> str:
        return (
            files("mcptune.synthesis")
            .joinpath("prompts", f"{self.prompt_version}.txt")
            .read_text(encoding="utf-8")
        )

    @staticmethod
    def _template_fallback(tool: ToolSpec, arguments: dict[str, Any]) -> str:
        if arguments:
            return f"Use the {tool.name} tool with arguments: {json.dumps(arguments)}"
        return f"Use the {tool.name} tool."

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."
