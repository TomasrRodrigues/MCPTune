# Configuration reference

Every `MCPTune(...)` parameter and config option, reconciled with the current API.

## `MCPTune(...)`

| Parameter         | Type                     | Default   | Description                                                          |
|-------------------|--------------------------|-----------|----------------------------------------------------------------------|
| `model`           | `str`                    | required  | Base model identifier passed to the trainer.                         |
| `mcpserver`       | any                      | required  | Target for the adapter (FastMCP in-memory, stdio path, or URL).      |
| `adapter`         | `MCPAdapter` \| None     | None      | Custom transport adapter; defaults to `FastMCPAdapter(mcpserver)`.   |
| `seed`            | `int` \| None            | 0         | Root seed for reproducible sampling.                                 |
| `llm_backend`     | `str`                    | `"none"`  | Shared default LLM backend for all generation stages. See below.     |
| `llm_model`       | `str` \| None            | None      | Shared model name for all LLM stages (e.g. `"qwen2.5:3b"`).          |
| `strict_llm`      | `bool`                   | False     | Raise on a configured-but-unreachable backend instead of falling back.|
| `semantic_backend`| `str` \| None            | None      | Per-stage override; inherits `llm_backend` (or `"local"`) if None.   |
| `intent_backend`  | `str` \| None            | None      | Per-stage override; inherits `llm_backend` if None.                  |
| `semantic_model`  | `str` \| None            | None      | Per-stage model override; inherits `llm_model` if None.              |
| `intent_model`    | `str` \| None            | None      | Per-stage model override; inherits `llm_model` if None.              |
| `intent_llm_call` | `Callable` \| None       | None      | Inject a custom LLM call (testing).                                  |
| `trainer`         | `TrainerBackend` \| None | None      | Required only to call `.train()`.                                    |

### LLM backend resolution

`llm_backend` / `llm_model` are the single knobs you normally set; the stages
inherit them. Per-stage overrides win when provided. Resolution:

- `semantic_backend = semantic_backend or (llm_backend if llm_backend != "none" else "local")`
- `intent_backend = intent_backend or llm_backend`
- models inherit `llm_model` unless a per-stage model is given.

So defaults (nothing set) keep semantic on the `local` lookup table and intent
on templates. `llm_backend="ollama"` drives every stage through Ollama with one
setting — including the semantic sampler, which previously had to be configured
separately.

Backend values: `"none"` (template/skip), `"local"` (semantic lookup table only),
`"ollama"` (local Ollama HTTP), `"transformers"` (in-process HF; needs the extra).

## Pipeline methods

| Call                                  | Sync/async | What it does                                              |
|---------------------------------------|------------|-----------------------------------------------------------|
| `await tuner.discover()`              | async      | Discover tools → `list[ToolSpec]`.                        |
| `tuner.build_dataset(tools, samples_per_tool=1)` | sync | Sample arguments + synthesize intents. **Offline.**   |
| `await tuner.execute(dataset)`        | async      | Run each tool (capture result) + synthesize final answer. |
| `tuner.train(dataset, config=None)`   | sync       | Fine-tune via the configured trainer.                     |

`build_dataset` is offline (no server/LLM calls beyond intent synthesis);
`execute` is the online pass that hits the server and the answer LLM. A dataset
without `execute()` still trains as 2-turn (call-only) data; running `execute()`
upgrades it to the full call → result → answer loop.

## Training config (`TransformersTrainerBackend`)

Passed via `tuner.train(dataset, config={...})`:

| Key             | Default      | Notes                                   |
|-----------------|--------------|-----------------------------------------|
| `format`        | `"tool_use"` | Training format (native tool-use).      |
| `lora_rank`     | `8`          |                                         |
| `lora_alpha`    | `16`         |                                         |
| `lora_dropout`  | `0.05`       |                                         |
| `learning_rate` | `2e-4`       |                                         |
| `epochs`        | `1`          |                                         |
| `batch_size`    | `1`          | Per-device.                             |
| `max_length`    | `512`        | Truncation; raise if examples are masked away. |

The base model's tokenizer must have a `chat_template` that supports `tools=`.

## Environment variables

| Variable      | Used by                       | Default                  |
|---------------|-------------------------------|--------------------------|
| `OLLAMA_HOST` | LLM backends (ollama)         | `http://localhost:11434` |