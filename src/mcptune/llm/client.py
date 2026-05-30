"""Shared LLM backend dispatcher for synthesis modules.

Two free, locally-runnable backends:
- "ollama" — HTTP to a local Ollama server (no API keys, no signup)
- "transformers" — HuggingFace transformers in-process (heavier dep)

Used by:
- mcptune.sampling.semantic.SemanticSampler
- mcptune.synthesis.intent.IntentSynthesizer (Issue 16 PR B)
"""

from __future__ import annotations

import os


class LLMClient:
    def __init__(
        self,
        backend: str,
        model: str | None = None,
        temperature: float = 0.0,
        ollama_host: str | None = None,
    ):
        if backend not in ("ollama", "transformers"):
            raise ValueError(
                f"Unknown LLM backend: {backend!r}. Expected 'ollama' or 'transformers'."
            )
        self.backend = backend
        self.model = model
        self.temperature = temperature
        self.ollama_host = ollama_host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        # Lazy-loaded transformers state — populated on first call.
        self._hf_tokenizer = None
        self._hf_model = None
        self._hf_model_name: str | None = None

    def generate(self, prompt: str, *, json_mode: bool = False) -> str:
        """Generate a response. `json_mode=True` requests structured JSON
        output where the backend supports it (Ollama does; transformers
        relies on prompt discipline)."""
        if self.backend == "ollama":
            return self._call_ollama(prompt, json_mode=json_mode)
        return self._call_transformers(prompt)

    def _call_ollama(self, prompt: str, *, json_mode: bool) -> str:
        import httpx  # transitive via fastmcp

        model = self.model or "qwen2.5:7b"
        body: dict = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": self.temperature},
        }
        if json_mode:
            body["format"] = "json"

        try:
            response = httpx.post(
                f"{self.ollama_host}/api/generate",
                json=body,
                timeout=120.0,
            )
            response.raise_for_status()
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Could not reach Ollama at {self.ollama_host}. "
                "Is `ollama serve` running? Install: https://ollama.com"
            ) from e
        return response.json().get("response", "")

    def _call_transformers(self, prompt: str) -> str:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Transformers backend requires the transformers and torch "
                "packages. Install with: pip install mcptune[transformers]"
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

        generated = outputs[0][inputs["input_ids"].shape[1] :]
        return self._hf_tokenizer.decode(generated, skip_special_tokens=True).strip()
