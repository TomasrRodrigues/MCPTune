from mcptune.schema.tools import ToolSpec


class HFIntentBackend:
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.model = None  # lazy load

    def _load(self):
        if self.model is None:
            from transformers import pipeline

            self.model = pipeline("text-generation", model=self.model_id)

    def generate_intent(self, tool: ToolSpec) -> dict:
        self._load()

        prompt = self._build_prompt(tool)

        out = self.model(prompt, temperature=0.0, max_new_tokens=128)[0]["generated_text"]

        return self._parse(out)
