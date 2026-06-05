"""Prove the fine-tune: base vs tuned tool-call accuracy on held-out intents.

Requires mcptune[transformers] and a trained adapter (run full_pipeline.py
first). Use a 1.5-3B model for a real result; 135M is below the floor.
"""

import asyncio

from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcptune.evaluation import EvalCase, evaluate, format_comparison
from mcptune.runtime import TransformersModelRunner

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = "./mcptune-demo-output/finetuned"

# Held out — NOT in the training set.
CASES = [
    EvalCase("What's the weather in Madrid?", "get_weather", {"city"}),
    EvalCase("Is it raining in Berlin right now?", "get_weather", {"city"}),
    EvalCase("Email alice@example.com to say the meeting moved to 3pm",
             "send_email", {"to", "subject", "body"}),
]


def _tools_from_server():
    from fastmcp import FastMCP
    from mcptune.adapters.fastmcp import FastMCPAdapter

    server = FastMCP("eval-demo")

    @server.tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Sunny in {city}"

    @server.tool
    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email."""
        return f"sent to {to}"

    return asyncio.run(FastMCPAdapter(server).discover_tools())


def main():
    tools = _tools_from_server()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)
    base = evaluate(TransformersModelRunner(base_model, tokenizer), CASES, tools)

    tuned_model = PeftModel.from_pretrained(
        AutoModelForCausalLM.from_pretrained(BASE_MODEL), ADAPTER_PATH
    )
    tuned = evaluate(TransformersModelRunner(tuned_model, tokenizer), CASES, tools)

    print(format_comparison(base, tuned))


if __name__ == "__main__":
    main()