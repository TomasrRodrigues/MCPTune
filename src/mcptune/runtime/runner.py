"""Model abstraction for the runtime.

The loop depends only on the ModelRunner protocol, so tests can inject a
scripted fake and eval can swap base vs fine-tuned models. The concrete
TransformersModelRunner wraps an HF model + tokenizer and applies the
chat template with tools=, so the inference prompt matches training.
"""

from __future__ import annotations

from typing import Any, Protocol


class TokenizedInputs(Protocol):
    def to(self, device: str) -> Any: ...

    def __getitem__(self, key: str) -> Any: ...


class TokenizerProtocol(Protocol):
    chat_template: str | None
    pad_token_id: int | None
    eos_token_id: int | None

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tools: list[dict[str, Any]] | None,
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str: ...

    def __call__(self, text: str, *, return_tensors: str) -> TokenizedInputs: ...

    def decode(self, token_ids: Any, *, skip_special_tokens: bool) -> str: ...


class ModelProtocol(Protocol):
    def generate(self, **kwargs: Any) -> Any: ...


class ModelRunner(Protocol):
    def generate(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str: ...


class TransformersModelRunner:
    def __init__(
        self,
        model: ModelProtocol,
        tokenizer: TokenizerProtocol,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        device: str | None = None,
    ):
        try:
            import torch
        except ImportError as e:
            raise ImportError(
                "TransformersModelRunner requires torch + transformers. "
                "Install with: pip install mcptune[transformers]"
            ) from e
        if tokenizer.chat_template is None:
            raise ValueError(
                "Tokenizer has no chat_template; the runtime needs one to "
                "render tools into context. Use a chat/instruct model."
            )
        self._torch = torch
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def generate(self, messages: list[dict[str, str]], tools: list[dict[str, Any]]) -> str:
        torch = self._torch
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools or None,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        }
        if self.temperature > 0:
            gen_kwargs.update(do_sample=True, temperature=self.temperature)
        else:
            gen_kwargs.update(do_sample=False)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        # NOTE: Qwen2.5 renders <tool_call> as literal text, so skip_special_tokens
        # is safe. If a model defines it as a special token, calls will vanish from
        # the decoded string — flip this to False for that model family.
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
