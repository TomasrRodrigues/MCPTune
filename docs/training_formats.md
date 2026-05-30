# Training format emitters

`mcptune.formats` converts `DatasetRow` collections into the message shapes that fine-tuning frameworks consume. Each emitter is a free function `list[DatasetRow] -> list[dict]`; the registry `mcptune.formats.FORMATS` maps a string identifier to its emitter for dynamic dispatch.

## Usage

```python
from mcptune.formats import openai_messages, convert

dataset = tuner.build_dataset(tools)

# Direct call when you know the format
openai_rows = openai_messages(dataset)

# Or via the registry
trl_rows = convert(dataset, "trl")
```

## Available formats

| Format key  | Emitter              | Typical consumer                                              |
|-------------|----------------------|---------------------------------------------------------------|
| `openai`    | `openai_messages`    | OpenAI fine-tuning API; OpenAI-compatible SDKs                |
| `sharegpt`  | `sharegpt`           | ShareGPT-format datasets; community fine-tuning recipes       |
| `trl`       | `trl_sft`            | HuggingFace TRL `SFTTrainer` with non-tool-aware chat templates |
| `anthropic` | `anthropic_tool_use` | Anthropic Messages API tool-use training data                 |

## Format details

### `openai`

Three-message conversation per row: `user` → `assistant` (with one `tool_calls` entry) → `tool`. The `tool_call_id` comes from `row.request["id"]` so it's unique across the dataset. Tool call `arguments` are JSON-encoded strings, per the OpenAI spec.

### `sharegpt`

Three-turn conversation per row in `{conversations: [{from, value}]}` shape. Roles are `human`, `gpt`, `tool`. The `gpt` turn's value is a JSON object containing a `tool_call` field with `name` and `arguments` - ShareGPT has no universal tool-calling convention, so we serialize the call as inline JSON. Adjust to match your fine-tuning recipe if needed.

### `trl`

Three-message conversation in `{messages: [{role, content}]}` shape with stringified content for every message. This is the conservative default for TRL `SFTTrainer` setups where the chat template does not natively handle structured tool calls. If you're on a newer tool-aware chat template (e.g., Qwen2's), the `openai` format works directly with TRL.

### `anthropic`

Three-turn conversation in Anthropic Messages API shape. Anthropic puts `tool_result` blocks back into a `user` turn - there is no separate `tool` role. The `id` field on the `tool_use` block matches `row.request["id"]`.

## User intents

If Issue 16 has shipped (it has - `IntentSynthesizer`), every row produced by `MCPTune.build_dataset` carries a `user_intent`. Emitters use that directly as the `user` message content.

If a `DatasetRow` is constructed manually without a `user_intent`, the emitters fall back to a deterministic template (`"Use the {tool_name} tool with arguments: ..."`). This is fine for ad-hoc use but produces lower-quality training data than synthesized intents.

## Adding a new format

1. Write `mcptune/formats/myformat.py` with a function
   `myformat(rows: list[DatasetRow]) -> list[dict]`.
2. Register it in `mcptune/formats/__init__.py`'s `FORMATS` dict.
3. Add tests in `tests/unit/test_formats.py` covering the output shape.
4. Document the format above.

The fallback for missing `user_intent` lives in `mcptune.formats._common.template_intent` - reuse it.