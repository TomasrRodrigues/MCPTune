# Configuration reference

Every constructor parameter, config dict, and environment variable MCPTune respects.

## `MCPTune(...)` parameters

| Parameter           | Type                    | Default     | Description                                                            |
|---------------------|-------------------------|-------------|------------------------------------------------------------------------|
| `model`             | `str`                   | (required)  | Base model identifier passed to the trainer. Not loaded by MCPTune itself. |
| `mcpserver`         | any                     | (required)  | Target for the adapter. FastMCP `Client` accepts in-memory, stdio path, or URL. |
| `adapter`           | `MCPAdapter` \| None    | None        | Custom transport adapter. Defaults to `FastMCPAdapter(mcpserver)`.     |
| `seed`              | `int` \| None           | 0           | Root seed for reproducible sampling. Same seed + same server = same dataset. |
| `semantic_backend`  | `str`                   | `"local"`   | Semantic sampler backend. See below.                                   |
| `intent_backend`    | `str`                   | `"none"`    | Intent synthesizer backend. See below.                                 |
| `intent_model`      | `str` \| None           | None        | Specific model identifier for the intent backend (e.g., `"qwen2.5:7b"`). |
| `intent_llm_call`   | `Callable` \| None      | None        | Inject a custom LLM call for testing.                                  |
| `trainer`           | `TrainerBackend` \| None | None       | Training backend. Required only if calling `mcp.train()`.              |

## Sampling

### `semantic_backend`

Controls how argument values are generated for known parameter shapes.

| Value           | Behavior                                                                  | Requires            |
|-----------------|---------------------------------------------------------------------------|---------------------|
| `"local"`       | Lookup table (city → "Lisbon", email → "test@example.com", etc.)         | nothing             |
| `"none"`        | Skip semantic generation; fall through to structural sampling             | nothing             |
| `"ollama"`      | Generate via local Ollama HTTP API                                        | `ollama serve` running |
| `"transformers"`| Generate via HuggingFace transformers in-process                          | `mcptune[transformers]` |

Composes with structural recursive sampling — semantic fills what it knows, recursive fills the rest. See [grounding.md](grounding.md) for prompt details.

### Sampling reproducibility

A given `(seed, tool_name, sample_index)` triple always produces the same arguments. Different `seed`s produce different datasets; different `sample_index`es within a seed produce diverse samples. Multi-sample runs (`samples_per_tool > 1`) are automatically diverse without being random.

## Intent synthesis

### `intent_backend`

Controls how the natural-language `user_intent` for each row is generated.

| Value            | Behavior                                                                  | Requires            |
|------------------|---------------------------------------------------------------------------|---------------------|
| `"none"`         | Templated fallback: `"Use the {tool_name} tool with arguments: {...}"`    | nothing             |
| `"ollama"`       | Generate via Ollama; falls back to template on error/empty                | `ollama serve` running |
| `"transformers"` | Generate via HuggingFace in-process                                       | `mcptune[transformers]` |

Each `DatasetRow` records `intent_prompt_version` — `None` when the template was used (either configured that way or fallen back to), the prompt version (`"intent_v1"`) when the LLM actually produced the intent. This keeps provenance honest.

## Training

### `MCPTune.train(dataset, config=None)`

Delegates to `self.trainer.train(self.model, dataset, config)`. Raises if no trainer was configured.

### `TransformersTrainerBackend` config

| Key             | Default     | Notes                                              |
|-----------------|-------------|----------------------------------------------------|
| `format`        | `"trl"`     | Training format from `mcptune.formats`             |
| `lora_rank`     | `8`         | LoRA rank (`r`)                                    |
| `lora_alpha`    | `16`        | LoRA alpha (scaling)                               |
| `lora_dropout`  | `0.05`      |                                                    |
| `learning_rate` | `2e-4`      | Higher than full fine-tune; LoRA tolerates this    |
| `epochs`        | `1`         |                                                    |
| `batch_size`    | `1`         | Per-device                                         |
| `max_length`    | `512`       | Token truncation                                   |
| `warmup_steps`  | `0`         |                                                    |
| `logging_steps` | `10`        |                                                    |
| `save_strategy` | `"epoch"`   |                                                    |

Pass via `tuner.train(dataset, config={...})` or `backend.train(model_name, dataset, config={...})`. The base model's tokenizer must have a `chat_template`.

## Environment variables

| Variable        | Used by                                  | Default                  |
|-----------------|------------------------------------------|--------------------------|
| `OLLAMA_HOST`   | `LLMClient` (semantic + intent backends) | `http://localhost:11434` |

## File and directory conventions

- Default training output directory: `./mcptune-checkpoints` (configurable via `TransformersTrainerBackend(output_dir=...)`).
- `write_jsonl` accepts `str | Path`; creates parent dirs as needed.
- Prompt templates live in `mcptune/sampling/prompts/` and `mcptune/synthesis/prompts/`, shipped as package data.

## Prompt versions

Bundled prompt versions are pinned for reproducibility. To use a different version, pass `prompt_version=...` to the relevant component:

- `SemanticSampler(prompt_version=...)`:
  - `"grounded_semantic_v1"` (default)
  - `"semantic_v1"` (ungrounded, for ablation)
- `IntentSynthesizer(prompt_version=...)`:
  - `"intent_v1"` (default)

A row's `intent_prompt_version` field records exactly which prompt produced its intent, so datasets generated with different prompt versions stay distinguishable.