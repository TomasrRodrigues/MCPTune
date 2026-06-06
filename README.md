# MCPTune

Fine-tune a small language model to call a specific MCP server's tools.

[![CI](https://github.com/YOUR_ORG/mcptune/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/mcptune/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

Give MCPTune an [MCP](https://modelcontextprotocol.io) server and a base model.
It discovers the server's tools, generates a synthetic tool-use dataset, and
fine-tunes the model to emit that server's tool calls in a format a runtime can
execute. A built-in minimal runtime and evaluation let you measure whether the
fine-tune actually worked.

The thesis MCPTune exists to test: **small models can be trained to do reliable
MCP tool-calling for a specific server, cheaply and locally.** This release is
the pipeline that produces and *measures* that - run `examples/evaluate.py` to
get the base-vs-tuned number for your server and model.

## Install

Not yet on PyPI - install from source:

```bash
git clone https://github.com/YOUR_ORG/mcptune
cd mcptune
pip install -e .                      # core: discovery, dataset generation, formats
pip install -e ".[transformers]"      # add LoRA fine-tuning + the runtime model runner
pip install -e ".[dev]"               # contributor tooling
```

<!-- Once published: pip install mcptune  /  pip install "mcptune[transformers]" -->

For realistic intents and arguments, run a local [Ollama](https://ollama.com)
(`ollama serve`, then `ollama pull qwen2.5:3b`). Without it, MCPTune falls back
to templated intents and a lookup table - fine for a smoke test, not for
training data you'd trust.

## How it works

```
discover → sample args → synthesize intent → execute → synthesize answer → train → evaluate
adapter    sampling        synthesis        runtime      synthesis       LoRA      runtime
```

The model only ever emits a tool-call *request* as text. A runtime parses it,
executes it against the MCP server, and feeds the result back. MCPTune trains
the model to emit those requests well; the orchestration runtime is a separate
concern (this repo ships a minimal one for evaluation and demos; production
deployments use an MCP-aware framework).

## Quickstart

```python
import asyncio
from fastmcp import FastMCP
from mcptune import MCPTune

server = FastMCP("demo")

@server.tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    return f"Sunny and 22C in {city}"

async def main():
    tuner = MCPTune(
        model="Qwen/Qwen2.5-1.5B-Instruct",
        mcpserver=server,
        llm_backend="ollama",        # natural intents + arguments (needs Ollama)
        llm_model="qwen2.5:3b",
        seed=42,
    )
    tools = await tuner.discover()
    dataset = tuner.build_dataset(tools, samples_per_tool=8)   # sample + intent (offline)
    dataset = await tuner.execute(dataset)                     # tool results + final answers

    for row in dataset[:3]:
        print(f"{row.user_intent}  ->  {row.tool_name}({row.arguments})")

asyncio.run(main())
```

Fine-tune on the dataset (needs `mcptune[transformers]`):

```python
from mcptune.training.backends.transformers_backend import TransformersTrainerBackend

tuner = MCPTune(
    model="Qwen/Qwen2.5-1.5B-Instruct",
    mcpserver=server,
    llm_backend="ollama", llm_model="qwen2.5:3b",
    trainer=TransformersTrainerBackend(output_dir="./checkpoints"),
    seed=42,
)
tools = await tuner.discover()
dataset = await tuner.execute(tuner.build_dataset(tools, samples_per_tool=20))
trained = tuner.train(dataset, config={"epochs": 1})
```

Then measure base vs tuned with `examples/evaluate.py`, or run the model against
your server through the minimal runtime with `examples/run_agent.py`.

## What's in 0.1.0

| Capability                                          | Status |
|-----------------------------------------------------|--------|
| Tool discovery + JSONSchema fidelity (FastMCP)      | ✅ |
| Structural + semantic argument sampling (seeded)    | ✅ |
| Intent synthesis (template / Ollama / transformers) | ✅ |
| Native tool-use training format + SFT masking       | ✅ |
| Tool execution + answer synthesis (full loop)       | ✅ |
| LoRA fine-tuning (transformers + PEFT)              | ✅ |
| Minimal runtime + before/after evaluation           | ✅ |
| stdio / HTTP adapters                               | 🚧 planned |
| Full evaluation pipeline, forward-mode generation   | 🚧 0.2.0 |
| Production runtime, multi-turn, personas            | 🚧 1.0.0 |

## Documentation

- [Quickstart](docs/quickstart.md)
- [Configuration reference](docs/configuration.md)
- [Training](docs/training.md) · [Training formats](docs/training_formats.md) · [Grounding](docs/grounding.md)
- [Examples](examples/)

## Supported models

Native tool-call emission targets **Qwen** (`<tool_call>`) first. Other open-weight
families with tool-aware chat templates (Llama 3.1+, Mistral) need a parser entry
in `mcptune.runtime.parsing`. MCPTune fine-tunes open-weight models only - it does
not train closed APIs.

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the workflow and architectural
constraints. Issues are tracked by milestone (0.1.x, 0.2.0, 1.0.0).

## License

MIT - see [`LICENSE`](LICENSE).