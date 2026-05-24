# MCPTune

Synthetic dataset generation and fine-tuning infrastructure for MCP-based tool use.

[![CI](https://github.com/YOUR_ORG/mcptune/actions/workflows/ci.yml/badge.svg)](https://github.com/YOUR_ORG/mcptune/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> ⚠️ **Early development.** APIs are unstable and may change between commits. Not yet on PyPI.

## What it does

MCPTune connects to an [MCP](https://modelcontextprotocol.io) server, discovers its tools, generates valid synthetic invocations from each tool's JSON schema, executes them against the server, and captures the request/response pairs as a training dataset for fine-tuning tool-using language models.

The long-term goal is one-line dataset generation:

​```python
from mcptune import MCPTune

tuner = MCPTune(model="base-model", mcpserver=my_server)
model, metrics = await tuner.run()
​```

## Status

| Capability | Status |
|------------|--------|
| FastMCP adapter | ✅ |
| Tool discovery + schema normalization | ✅ |
| Primitive argument sampling (string, int, float, bool) | ✅ |
| Closed-loop execution + response capture | ✅ |
| Recursive schema sampling (nested, arrays, enums) | 🚧 planned |
| Semantic argument generation | 🚧 planned |
| HTTP / stdio adapters | 🚧 planned |
| Training-format emission | 🚧 planned |
| Evaluation pipeline | 🚧 planned |

## Quickstart

​```bash
git clone https://github.com/TomasrRodrigues/mcptune
cd mcptune
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
​```

A minimal end-to-end run against an in-memory FastMCP server:

​```python
import asyncio
from fastmcp import FastMCP
from mcptune import MCPTune

server = FastMCP("demo")

@server.tool
def get_weather(city: str) -> str:
    return f"Sunny in {city}"

async def main():
    tuner = MCPTune(model="demo-model", mcpserver=server, seed=42)
    tools = await tuner.discover()
    dataset = tuner.build_dataset(tools)
    for row in dataset:
        print(row)

asyncio.run(main())
​```

Pass `seed=` when you need repeatable primitive/recursive sampler output for regression tests or dataset versioning. LLM-backed samplers are best-effort reproducible because providers may still be non-deterministic even when temperature and provider seeds are fixed.

## How it's built

​```
MCP Server  →  Adapter  →  Discovery  →  Sampler  →  Dataset Builder  →  Executor
​```

Each layer is a single-purpose module behind an interface. Transports go through adapters (`mcptune.adapters`), argument generation through samplers (`mcptune.sampling`), and orchestration lives in `mcptune.MCPTune`. The internal representation is `ToolSpec` - everything downstream of discovery operates on it, never on transport-specific types.

See [CONTRIBUTING.md](CONTRIBUTING.md) for architecture details and extension points.

## License

MIT. See [LICENSE](LICENSE).