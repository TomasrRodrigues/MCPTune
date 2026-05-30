# Contributing to MCPTune

Thanks for your interest. MCPTune is in active early development; contributions, bug reports, and architectural feedback are all welcome.

This document covers the practical workflow. For project goals and roadmap, see [README.md](README.md).

## Local development setup

Requires Python 3.10 or newer.

​```bash
git clone https://github.com/YOUR_ORG/mcptune
cd mcptune
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
​```

The `[dev]` extra pulls everything needed for development: pytest, pytest-asyncio, coverage, hypothesis, ruff, mypy, and fastmcp.

## Running tests

Tests are organized in three layers, distinguished by pytest markers:

​```bash
pytest                    # unit + integration (e2e excluded by default)
pytest -m unit            # fast, no IO - the loop you want during dev
pytest -m integration     # adapter ↔ core, in-process MCP servers
pytest -m e2e             # spawns external processes; opt-in only
​```

With coverage:

​```bash
pytest --cov=mcptune --cov-report=term
​```

When you add a feature, write the test alongside the code. When you fix a bug, add the regression test first.

## Code style

Ruff handles both linting and formatting. Before committing:

​```bash
ruff format src tests
ruff check src tests
​```

CI runs both as required checks. Configuration lives in `pyproject.toml`; the active ruleset is intentionally lean (`E`, `F`, `I`, `B`, `UP`) to keep contributor friction low.

## Commit conventions

Use [conventional commits](https://www.conventionalcommits.org/) for the subject line:

​```
<type>(<scope>): <imperative summary>
​```

Common types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`, `ci`. Scope is the affected module - `adapters`, `sampling`, `schema`, etc.

Wrap the body at ~72 characters and explain *why* the change matters. Reference issues by number when applicable.

## Pull request workflow

1. Open a draft PR early if the change is non-trivial - architecture conversations are easier with code than in the abstract.
2. Keep PRs small and single-purpose. A test refactor and a feature addition belong in separate PRs.
3. Make sure CI is green before requesting review.
4. Squash-and-merge is the default. The history on `main` should read as a logical sequence of changes, not your working history.

## Architecture quick reference

Before changing core code, please skim these constraints - they shape what reviewers will ask for.

**Transports stay behind adapters.** Anything that knows about JSON-RPC envelopes, HTTP, stdio, or FastMCP types belongs in `mcptune/adapters/`. Core code operates only on `ToolSpec` and the normalized response dict.

**Schema fidelity is non-negotiable.** When normalizing a tool, the full original JSONSchema must reach `ToolSpec.raw_input_schema` intact. Any normalization that loses or transforms schema information is a regression even if tests pass.

**Adapters share a contract.** Every adapter implements `discover_tools()` and `call_tool()` with identical return shapes. The contract is exercised in `tests/integration/test_fastmcp_adapter.py` - when you add an adapter, the same assertions should hold against it.

**Samplers are schema-driven, not domain-driven.** No hardcoded knowledge about specific tool names or argument meanings. Generation logic decides based on the JSONSchema, not the parameter name. Semantic generation (planned) will operate via metadata, not lookup tables.

## Extension points

The two places where most contributor PRs will land:

**Adding a new transport adapter.** Subclass `MCPAdapter` (`mcptune/adapters/base.py`), implement `discover_tools` and `call_tool`, and add tests that mirror the FastMCP adapter's contract test. Don't import internals of any specific MCP library; if your adapter speaks HTTP, talk to the server through a real HTTP client.

**Adding a sampler strategy.** Subclass `ArgumentSampler` (`mcptune/sampling/base.py`). Today: `PrimitiveSampler` handles flat primitive schemas. Likely next additions: recursive sampling for nested/array/enum schemas, then semantic sampling for schema-conditioned LLM generation.

## Asking questions

For architectural proposals or unclear-scope discussions, open an issue using the *Architectural discussion* template. For concrete bugs and feature requests, the standard templates work better.

Thanks for contributing.