# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — first end-to-end release

The full pipeline runs: discover MCP tools, generate synthetic datasets, emit them in standard training formats, and fine-tune a base model with LoRA.

### Added

- FastMCP transport adapter with schema-fidelity guarantees.
- Tool discovery normalizing into `ToolSpec` / `ToolParameter`.
- `PrimitiveSampler` for flat schemas.
- `RecursiveSampler` for nested objects, arrays, enums, `anyOf` / `oneOf`, nullable unions, and JSONSchema constraints (`minimum`, `maximum`, `minLength`, `maxLength`, `pattern`, common `format` values).
- Deterministic seeded sampling with per-(tool, sample) RNG so multi-sample runs are diverse but reproducible.
- `SemanticSampler` with four backends: `local` (lookup table), `none`, `ollama`, `transformers`. Tool description grounding via the `grounded_semantic_v1` prompt template.
- `IntentSynthesizer` with template fallback and `ollama` / `transformers` LLM backends. Per-row provenance via `intent_prompt_version`.
- `DatasetRow` schema with execution capture (`response`, `error`) and intent fields. JSONL persistence with `schema_version: 1` injected at write time, validation on read.
- Closed-loop execution: `FastMCPAdapter.call_tool` normalizes responses into `{content, structured_content, is_error}`.
- Training-format emitters: `openai_messages`, `sharegpt`, `trl_sft`, `anthropic_tool_use`, plus a `convert(rows, format)` registry.
- `TrainerBackend` ABC with two reference implementations:
  - `TransformersTrainerBackend` — real LoRA fine-tuning via transformers + PEFT.
  - `MockTrainerBackend` — no-op backend for fast unit tests.
- 183-test suite covering unit, integration, and e2e layers. Property-based tests for samplers via Hypothesis.

### Known limitations

- FastMCP is the only transport adapter; stdio and HTTP are planned.
- No evaluation pipeline yet (Phase 5 on the roadmap).
- No CLI; library only.
- Training is single-GPU / CPU, LoRA only. QLoRA, full fine-tune, DPO, RLHF are planned.
- A `TransformersTrainerBackend` instance trains one model at a time. For concurrent training, use separate instances.
- `transformers` is pinned to `>=4.40,<5.0` because the `Trainer` API renamed `tokenizer=` to `processing_class=` in 5.x. The upgrade lands in a later release.

### Dependencies

- Core: `fastmcp>=2.0`.
- Training extra: `transformers>=4.40,<5.0`, `torch>=2.0`,   `peft>=0.10`, `accelerate>=0.30`, `datasets>=2.18`.
- Dev: `pytest`, `pytest-asyncio`, `pytest-cov`, `hypothesis`,   `ruff`, `mypy`.

[0.1.0]: https://github.com/TomasrRodrigues/mcptune/releases/tag/v0.1.0