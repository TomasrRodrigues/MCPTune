# MCPTune

Synthetic dataset generation and fine-tuning infrastructure for MCP-based
tool use.

[![CI](https://github.com/TomasrRodrigues/mcptune/actions/workflows/ci.yml/badge.svg)](https://github.com/TomasrRodrigues/mcptune/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

MCPTune connects to an [MCP](https://modelcontextprotocol.io) server, discovers its tools, generates valid synthetic invocations from each tool's JSON schema, synthesizes natural-language user prompts that would elicit those calls and emits the results as a training dataset for fine-tuning tool-using language models. A built-in LoRA training backend closes the loop.


## Install

```bash
# Base install - dataset generation, all four training-format emitters
pip install mcptune

# Add real LoRA fine-tuning (heavy: pulls torch, transformers, peft)
pip install "mcptune[transformers]"

# Contributor / dev install
pip install -e ".[dev]"
```

Not yet on PyPI; for now install from source:

```bash
git clone https://github.com/TomasrRodrigues/mcptune
cd mcptune
pip install -e .
```

## Quickstart

```python
import asyncio
from fastmcp import FastMCP
from mcptune import MCPTune

server = FastMCP("demo")

@server.tool
def get_weather(city: str) -> str:
    return f"Sunny in {city}"

@server.tool
def add(a: int, b: int) -> int:
    return a + b

async def main():
    tuner = MCPTune(model="demo-model", mcpserver=server, seed=42)
    tools = await tuner.discover()
    dataset = tuner.build_dataset(tools, samples_per_tool=3)

    for row in dataset:
        print(f"{row.tool_name}({row.arguments}) - intent: {row.user_intent}")

asyncio.run(main())
```

Output:
```
add({'a': 53, 'b': 12}) - intent: Use the add tool with arguments: {"a": 53, "b": 12}
add({'a': 7, 'b': 91}) - intent: ...
get_weather({'city': 'Lisbon'}) - intent: ...
...
```

For a fuller walkthrough that adds LLM-backed intent synthesis, JSONL persistence, format conversion and fine-tuning, see [`docs/quickstart.md`](docs/quickstart.md).



## What's in 0.1.0

| Capability                                          | Status |
|-----------------------------------------------------|--------|
| FastMCP transport adapter                           | ✅     |
| Tool discovery + full JSONSchema fidelity           | ✅     |
| Primitive + recursive schema sampling               | ✅     |
| Seeded / reproducible sampling                      | ✅     |
| Semantic argument generation (local + LLM backends) | ✅     |
| User intent synthesis with prompt-version provenance| ✅     |
| Dataset persistence to JSONL                        | ✅     |
| Closed-loop execution + response capture            | ✅     |
| Training-format emitters: OpenAI, ShareGPT, TRL, Anthropic | ✅ |
| LoRA fine-tuning via transformers + PEFT            | ✅     |
| HTTP / stdio adapters                               | 🚧 planned |
| Evaluation pipeline                                 | 🚧 planned |
| CLI                                                 | 🚧 planned |

## Architecture

```
MCP Server -> Adapter -> Discovery -> Sampler -> Intent Synthesizer
|
v
Dataset Builder -> JSONL
|
v
Format Emitter -> Trainer
```

Each layer is a single-purpose module behind an interface:

- **Transports** stay behind adapters (`mcptune.adapters`).
- **Argument generation** stays in samplers (`mcptune.sampling`).
- **Intent synthesis** stays in `mcptune.synthesis`.
- **Output shape** stays in `mcptune.formats`.
- **Training** stays in `mcptune.training`, behind a pluggable backend interface.

Everything downstream of discovery operates on `ToolSpec` and `DatasetRow`,
never on transport-specific types. See [`CONTRIBUTING.md`](CONTRIBUTING.md)
for architectural detail and extension points.

## Documentation

- [Quickstart](docs/quickstart.md) - full pipeline walkthrough
- [Configuration reference](docs/configuration.md) - every constructor parameter and config dict
- [Training](docs/training.md) - fine-tuning with LoRA, custom backends
- [Training formats](docs/training_formats.md) - OpenAI, ShareGPT, TRL, Anthropic emitters
- [Semantic grounding](docs/grounding.md) - how tool descriptions are used for argument synthesis
- [Examples](examples/) - runnable scripts

## Contributing

See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the contributor workflow, code
style, and architectural constraints reviewers will hold PRs to.

## License

MIT. See [`LICENSE`](LICENSE).
