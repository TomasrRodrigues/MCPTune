"""Synthesize the assistant's final natural-language answer from a tool result.

Mirrors IntentSynthesizer: same backends, same template-fallback +
provenance contract (prompt_version is None when the fallback ran).
Given the user's intent, the call, and the tool's result, produce the
reply the assistant would give — the turn that teaches the model to USE
a result, not just request one.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from importlib.resources import files
from typing import Any

from mcptune.llm.client import LLMClient
from mcptune.schema.tools import ToolSpec


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    prompt_version: str | None


class AnswerSynthesizer:
    # src/mcptune/synthesis/answer.py
    def __init__(
        self,
        backend: str = "none",
        prompt_version: str = "answer_v1",
        model: str | None = None,
        temperature: float = 0.7,
        llm_call: Callable[[str], str] | None = None,
        ollama_host: str | None = None,
        strict: bool = False,
    ):
        self.backend = backend
        self.prompt_version = prompt_version
        self.model = model
        self.temperature = temperature
        self.ollama_host = ollama_host
        self._llm_call = llm_call
        self._client: LLMClient | None = None
        self.strict = strict
        self._fallback_count = 0
        self._warned = False

    def synthesize(
        self, tool: ToolSpec, arguments: dict[str, Any], result_text: str
    ) -> AnswerResult:
        if self.backend == "none":
            return AnswerResult(self._template_fallback(tool, result_text), None)
        try:
            prompt = self._build_prompt(tool, arguments, result_text)
            response = self._call_llm(prompt)
        except Exception as e:
            if self.strict:
                raise RuntimeError(
                    f"Answer backend {self.backend!r} failed: {e}. Fix the backend, "
                    "or set strict_llm=False to fall back to templates."
                ) from e
            self._note_fallback(str(e))
            return AnswerResult(self._template_fallback(tool, result_text), None)

        text = response.strip()
        if not text:
            self._note_fallback("empty response")
            return AnswerResult(self._template_fallback(tool, result_text), None)
        return AnswerResult(text, self.prompt_version)

    def _note_fallback(self, reason: str) -> None:
        self._fallback_count += 1
        if not self._warned:
            self._warned = True
            print(
                f"[MCPTune Warn] Answer synthesis fell back to template "
                f"(backend={self.backend!r}): {reason}. Remaining fallbacks are "
                "counted and summarized at the end of execution."
            )

    @property
    def fallback_count(self) -> int:
        return self._fallback_count

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

    def _build_prompt(self, tool: ToolSpec, arguments: dict[str, Any], result_text: str) -> str:
        template = (
            files("mcptune.synthesis")
            .joinpath("prompts", f"{self.prompt_version}.txt")
            .read_text(encoding="utf-8")
        )
        return (
            template.replace("{tool_name}", tool.name)
            .replace("{tool_description}", (tool.description or "(no description)")[:800])
            .replace("{arguments_json}", json.dumps(arguments, indent=2))
            .replace("{result_text}", result_text[:1200])
        )

    @staticmethod
    def _template_fallback(tool: ToolSpec, result_text: str) -> str:
        snippet = result_text.strip()
        if snippet:
            return f"The {tool.name} call returned: {snippet}"
        return f"I've completed the {tool.name} request."
