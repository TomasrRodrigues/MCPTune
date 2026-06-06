# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project aims to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - tool-use fine-tuning pipeline

Point MCPTune at an MCP server and a base model. It generates a synthetic tool-use dataset, fine-tunes the model to emit that server's tool calls and measures whether it learned them.

### Added
- **Tool discovery** behind a transport-agnostic adapter (FastMCP), normalizing every tool into a `ToolSpec` with full JSONSchema fidelity.
- **Argument sampling**: structural (recursive, schema-driven, seeded and reproducible) plus semantic generation (lookup table or LLM-backed) for realistic values.
- **Intent synthesis**: natural-language user requests for each call, via a local template or an LLM backend, with prompt-version provenance.
- **Native tool-use training format** (`tool_use`): tool definitions in context plus the assistant call in the model's native format, rendered by the tokenizer chat template (Qwen first) so a trained model's output is parseable by a runtime. Round-trip verified (emit → template → parse).
- **call → result → answer conversations**: `MCPTune.execute()` runs each tool to capture a real result and synthesizes the assistant's final answer, so the model learns the full loop, not just call emission.
- **SFT label masking**: loss is computed only on assistant turns (call and answer), not on tool context, user turns, or padding.
- **`mcptune.runtime`**: a minimal agent loop (tools in context → parse call → execute via the adapter → feed result back), used for evaluation and as a runnable demo. Not a production runtime.
- **`mcptune.evaluation`**: minimal before/after scoring - base vs fine-tuned tool-call accuracy on held-out intents.
- **LoRA fine-tuning** via transformers + PEFT (`mcptune[transformers]`), behind a pluggable `TrainerBackend` interface (mock backend included for tests).
- **Unified LLM config**: one `llm_backend` / `llm_model` drives semantic sampling, intent synthesis, and answer synthesis, with per-stage overrides.
- **Fail-loud** on a configured-but-unreachable LLM backend (`strict_llm`), or warn-once plus an end-of-run fallback summary otherwise.
- Dataset persistence to JSONL with schema validation; four additional raw conversation emitters (`openai`, `sharegpt`, `trl`, `anthropic`).

### Notes
- Qwen is the supported model family for the native call format. Other families need a parser entry in `mcptune.runtime.parsing`.
- The orchestration runtime shipped here is intentionally minimal. For production, run the fine-tuned model inside an MCP-aware framework (the MCP SDK, LangChain, etc.). A production runtime is on the 1.0.0 roadmap.
- This release provides the pipeline to **measure** whether a small model can be fine-tuned to call a server's tools. The headline result is produced by running `examples/evaluate.py`; it is not asserted here.

### Known limitations / roadmap
- **0.1.1**: richer argument diversity.
- **0.2.0**: full evaluation pipeline (splits, metrics, CI gating), forward-mode dataset generation, transformers 5.x.
- **1.0.0**: production runtime, multi-turn dialogue, persona conditioning.

[0.1.0]: https://github.com/YOUR_ORG/mcptune/releases/tag/v0.1.0