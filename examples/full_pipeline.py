"""Full MCPTune pipeline: discover, generate, persist, format, train.

This example exercises every layer of the library. It does NOT require
Ollama (uses local lookup + template intents), but it DOES require
mcptune[transformers] for the training step and will download a small
base model the first time it runs.

Usage:
    pip install "mcptune[transformers]"
    python examples/full_pipeline.py
"""

import asyncio
from pathlib import Path

from fastmcp import FastMCP

from mcptune import MCPTune
from mcptune.dataset.io import read_jsonl, write_jsonl
from mcptune.formats import openai_messages
from mcptune.training.backends.transformers_backend import (
    TransformersTrainerBackend,
)


def build_server() -> FastMCP:
    server = FastMCP("full-pipeline-demo")

    @server.tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Sunny and 22C in {city}"

    @server.tool
    def send_email(to: str, subject: str, body: str) -> str:
        """Send an email."""
        return f"Email sent to {to}"

    return server


async def main() -> None:
    artifacts = Path("./mcptune-demo-output")
    artifacts.mkdir(exist_ok=True)

    # --- Stage 1: build the pipeline ----------------------------------------

    server = build_server()
    trainer = TransformersTrainerBackend(output_dir=str(artifacts / "checkpoints"))

    tuner = MCPTune(
        model="HuggingFaceTB/SmolLM-135M-Instruct",
        mcpserver=server,
        seed=42,
        trainer=trainer,
    )

    # --- Stage 2: discover and generate -------------------------------------

    tools = await tuner.discover()
    print(f"[1/5] Discovered {len(tools)} tools: {[t.name for t in tools]}")

    dataset = tuner.build_dataset(tools, samples_per_tool=8)
    print(f"[2/5] Generated {len(dataset)} synthetic rows")

    # --- Stage 3: persist ---------------------------------------------------

    jsonl_path = artifacts / "dataset.jsonl"
    write_jsonl(dataset, jsonl_path)
    print(f"[3/5] Wrote dataset to {jsonl_path}")

    reloaded = read_jsonl(jsonl_path)
    assert reloaded == dataset, "round-trip mismatch"

    # --- Stage 4: convert to a training format -----------------------------

    openai_rows = openai_messages(dataset)
    print(f"[4/5] Converted to OpenAI Chat Completions format")
    print(f"      First user message: {openai_rows[0]['messages'][0]['content']!r}")

    # --- Stage 5: train -----------------------------------------------------

    print("[5/5] Training (this will download SmolLM-135M on first run)...")
    trained = tuner.train(
        dataset,
        config={
            "epochs": 1,
            "lora_rank": 4,
            "max_length": 128,
            "batch_size": 1,
        },
    )

    save_path = artifacts / "finetuned"
    trainer.save(trained, str(save_path))
    print(f"      Adapter saved to {save_path}")
    print(f"      Base model: {trained.metadata['base_model']}")
    print(f"      Examples:   {trained.metadata['num_examples']}")

    print("\nDone. Artifacts in", artifacts.resolve())


if __name__ == "__main__":
    asyncio.run(main())