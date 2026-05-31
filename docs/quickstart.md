# Quickstart

This walks through the full MCPTune pipeline against a small in-process MCP server, end-to-end. By the end you'll have a JSONL dataset of synthetic tool calls, the same dataset converted to a training format, and a fine-tuned model on disk.

## Prerequisites

- Python 3.10 or newer
- The base install for everything except fine-tuning
- `mcptune[transformers]` for the fine-tuning section

```bash
pip install mcptune              # core
pip install "mcptune[transformers]"  # for the training step
```

Optional but useful:
- [Ollama](https://ollama.com) running locally if you want LLM-backed semantic sampling and intent synthesis. The pipeline works fine without it — it'll fall back to a lookup table and a templated intent.

## 1. Define an MCP server

For this guide we'll use a small in-process FastMCP server. In real use you'd point at any existing MCP server — local stdio binary, remote HTTP, or in-memory.

```python
from fastmcp import FastMCP

server = FastMCP("weather-demo")

@server.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny and 22C in {city}"

@server.tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    return f"Email sent to {to}"
```

## 2. Discover the tools

```python
import asyncio
from mcptune import MCPTune

tuner = MCPTune(
    model="HuggingFaceTB/SmolLM-135M-Instruct",
    mcpserver=server,
    seed=42,
)

async def discover():
    tools = await tuner.discover()
    for tool in tools:
        print(f"{tool.name}: {tool.description}")
        for p in tool.parameters:
            print(f"  - {p.name} ({p.schema.get('type')}) required={p.required}")

asyncio.run(discover())
```

`tuner.discover()` returns `list[ToolSpec]`. Every parameter's full JSONSchema is preserved in `tool.raw_input_schema` for downstream use.

## 3. Generate a synthetic dataset

```python
async def generate():
    tools = await tuner.discover()
    dataset = tuner.build_dataset(tools, samples_per_tool=5)
    for row in dataset[:3]:
        print(row.tool_name, row.arguments)

asyncio.run(generate())
```

Each `DatasetRow` carries:

- `tool_name` — which tool the row exercises
- `arguments` — the sampled arguments (schema-valid)
- `request` — the full JSON-RPC envelope ready to send to the server
- `user_intent` — a natural-language prompt that would elicit this call
- `intent_prompt_version` — provenance for the intent (None if the
  template fallback was used)
- `response` / `error` — populated after execution (next step)

The dataset is deterministic given a seed — re-running with the same `seed=42` produces the same dataset.

## 4. (Optional) Use real LLMs for grounded synthesis

The default backends are dependency-free:
- Semantic argument generation uses a local lookup table (city → "Lisbon", email → "test@example.com", etc.).
- Intent synthesis uses a templated fallback.

To get genuinely realistic data, point both at a local Ollama install:

```bash
ollama serve            # in another terminal
ollama pull qwen2.5:7b
```

```python
tuner = MCPTune(
    model="HuggingFaceTB/SmolLM-135M-Instruct",
    mcpserver=server,
    seed=42,
    semantic_backend="ollama",       # was "local"
    intent_backend="ollama",         # was "none"
    intent_model="qwen2.5:7b",
)
```

Now `arguments` will look like `{"to": "alice@example.com", "subject": "Lunch?", "body": "Are you free Thursday?"}` instead of lookup-table defaults, and `user_intent` will be a natural sentence like `"Send Alice an email asking if she's free for lunch Thursday."` rather than the templated form.

See [`docs/grounding.md`](grounding.md) for what metadata each backend sees and how to control it.

## 5. Persist to JSONL

```python
from mcptune.dataset.io import write_jsonl, read_jsonl

write_jsonl(dataset, "weather.jsonl")
loaded = read_jsonl("weather.jsonl")
assert loaded == dataset
```

The on-disk format is one JSON object per line. Each record includes a `schema_version: 1` field for future migrations. Validation is automatic on read — corrupt rows raise `DatasetValidationError`. See [`docs/training_formats.md`](training_formats.md) for the field-level schema.

## 6. Convert to a training format

```python
from mcptune.formats import openai_messages, sharegpt, trl_sft, anthropic_tool_use, convert

# Direct call
openai_rows = openai_messages(dataset)

# Or via the registry
trl_rows = convert(dataset, "trl")

print(openai_rows[0])
# {
#   "messages": [
#     {"role": "user", "content": "What's the weather in Lisbon?"},
#     {"role": "assistant", "tool_calls": [{...}]},
#     {"role": "tool", "tool_call_id": "...", "content": "Sunny and 22C in Lisbon"}
#   ]
# }
```

Four formats ship: `openai`, `sharegpt`, `trl`, `anthropic`. Pick the one your fine-tuning framework expects. See [`docs/training_formats.md`](training_formats.md) for details.

## 7. Fine-tune a model

```python
from mcptune.training.backends.transformers_backend import TransformersTrainerBackend

trainer = TransformersTrainerBackend(output_dir="./checkpoints")

tuner = MCPTune(
    model="HuggingFaceTB/SmolLM-135M-Instruct",
    mcpserver=server,
    seed=42,
    trainer=trainer,
)

async def full_pipeline():
    tools = await tuner.discover()
    dataset = tuner.build_dataset(tools, samples_per_tool=20)
    trained = tuner.train(dataset, config={"epochs": 1, "lora_rank": 8})
    trainer.save(trained, "./my-finetuned-model")

asyncio.run(full_pipeline())
```

`save()` writes the LoRA adapter (not the full base model) plus the tokenizer. To reload at inference:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
tokenizer = AutoTokenizer.from_pretrained("./my-finetuned-model")
model = PeftModel.from_pretrained(base, "./my-finetuned-model")
```

See [`docs/training.md`](training.md) for the full config dict, backend requirements, and how to write a custom backend.

## What's next?

- Run the full example end-to-end: [`examples/full_pipeline.py`](../examples/full_pipeline.py)
- Configure each layer in detail: [`docs/configuration.md`](configuration.md)
- Point at a real MCP server (FastMCP file, stdio binary, HTTP): the FastMCP `Client` already accepts all three — pass it as `mcpserver=...`.
- Contribute: [`CONTRIBUTING.md`](../CONTRIBUTING.md) lists the   open issues and architectural constraints.