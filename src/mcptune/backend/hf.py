"""HuggingFace-backed intent generation backend.

This module provides a simple wrapper around the HuggingFace
transformers `pipeline` for text-generation. It is used by the
intent synthesizer to generate user intent descriptions for a given
tool specification.

The implementation is intentionally minimal and performs a lazy load
of the HF pipeline to avoid importing heavy dependencies at module
import time.
"""

from mcptune.schema.tools import ToolSpec


class HFIntentBackend:
    """Intent generation using a HuggingFace text-generation pipeline.

    Parameters
    ----------
    model_id:
        The HF model identifier passed to `transformers.pipeline`.
    """

    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None  # lazy load

    def _load(self):
        """Lazily initialize the HF `pipeline` instance.

        This defers the heavy import of `transformers` until the first
        call to `generate_intent` so importing mcptune remains lightweight.
        """
        if self.model is None:
            from transformers import pipeline

            self.model = pipeline("text-generation", model=self.model_id)

    def generate_intent(self, tool: ToolSpec) -> dict:
        """Generate a natural-language intent for `tool`.

        Parameters
        ----------
        tool:
            The tool specification for which to generate an intent.

        Returns
        -------
        dict
            Parsed intent information as returned by the backend's
            `_parse` implementation (implementation-specific shape).
        """
        self._load()

        prompt = self._build_prompt(tool)

        out = self.model(prompt, temperature=0.0, max_new_tokens=128)[0]["generated_text"]

        return self._parse(out)
