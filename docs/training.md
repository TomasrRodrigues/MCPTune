# Training

MCPTune's training layer turns a `list[DatasetRow]` into a fine-tuned
model. The pipeline:

1. Generate a dataset via `MCPTune.build_dataset(...)` or load one from
   JSONL with `mcptune.dataset.io.read_jsonl(...)`.
2. Pick a `TrainerBackend` and pass it to `MCPTune`.
3. Call `mcp.train(dataset)` to run training.
4. Call `backend.save(trained_model, path)` to persist.

## Architecture

`mcptune.training.base.TrainerBackend` is an ABC with two methods:

```python
class TrainerBackend(ABC):
    @abstractmethod
    def train(self, model_name, dataset, config) -> TrainedModel: ...

    @abstractmethod
    def save(self, model: TrainedModel, path: str) -> None: ...
```

Anything that implements this contract plugs into `MCPTune`.

## Built-in backends

### `TransformersTrainerBackend` - real LoRA fine-tuning

HuggingFace transformers + PEFT LoRA. Requires `mcptune[transformers]`:

```bash
pip install mcptune[transformers]
```

Example end-to-end run:

```python
import asyncio
from fastmcp import FastMCP
from mcptune import MCPTune
from mcptune.training.backends.transformers_backend import (
    TransformersTrainerBackend,
)

server = FastMCP("demo")

@server.tool
def get_weather(city: str) -> str:
    return f"Sunny in {city}"

trainer = TransformersTrainerBackend(output_dir="./checkpoints")

tuner = MCPTune(
    model="HuggingFaceTB/SmolLM-135M-Instruct",
    mcpserver=server,
    intent_backend="ollama",       # synthesize realistic user prompts
    trainer=trainer,
)

async def main():
    tools = await tuner.discover()
    rows = tuner.build_dataset(tools, samples_per_tool=20)
    trained = tuner.train(rows, config={"epochs": 3, "lora_rank": 8})
    trainer.save(trained, "./my-finetuned-model")

asyncio.run(main())
```

#### Config dict

All keys are optional; defaults are tuned for small models.

| Key             | Default     | Notes                                              |
|-----------------|-------------|----------------------------------------------------|
| `format`        | `"trl"`     | Format from `mcptune.formats`                      |
| `lora_rank`     | `8`         |                                                    |
| `lora_alpha`    | `16`        |                                                    |
| `lora_dropout`  | `0.05`      |                                                    |
| `learning_rate` | `2e-4`      | LoRA tolerates higher rates than full fine-tune    |
| `epochs`        | `1`         |                                                    |
| `batch_size`    | `1`         | Per-device                                         |
| `max_length`    | `512`       | Token truncation                                   |
| `warmup_steps`  | `0`         |                                                    |
| `logging_steps` | `10`        |                                                    |
| `save_strategy` | `"epoch"`   |                                                    |

#### Base model requirements

The tokenizer must have a `chat_template`. Qwen2.x, Llama 3, Mistral
Instruct, and SmolLM-Instruct all qualify. Models without one (raw
GPT-2, base Llama 3 without `-Instruct`, etc.) raise `ValueError` at
training time - set `tokenizer.chat_template` manually before
`train()` if you really need to use one.

#### What gets saved

`save()` writes the PEFT adapter (`adapter_config.json` +
`adapter_model.safetensors`) and the tokenizer. The base model is NOT
saved - load it from HuggingFace Hub and apply the adapter at
inference time:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM-135M-Instruct")
tokenizer = AutoTokenizer.from_pretrained("./my-finetuned-model")
model = PeftModel.from_pretrained(base, "./my-finetuned-model")
```

### `MockTrainerBackend` - no-op for tests

Records `model_name`, `num_examples`, and `config` in metadata; never
actually trains. Useful for testing orchestration without heavy deps.

## Writing a custom backend

Subclass `TrainerBackend`. The contract:

- `train(model_name, dataset, config)` accepts `list[DatasetRow]` and
  returns a `TrainedModel` whose `backend` field identifies the
  implementation.
- `save(model, path)` persists whatever artifacts a user needs to
  reload the trained model later. Sets `model.model_path`.
- `train` then `save` is the expected call order on a single backend
  instance. `save` before `train` should raise.

## Limitations (v0.1)

- Single-GPU / CPU only.
- LoRA only via `TransformersTrainerBackend`. QLoRA, full fine-tune,
  DPO, and RLHF are future issues.
- A backend instance trains one model at a time. For concurrent
  training, use separate instances.
- No automatic train/eval split. The evaluation pipeline (Phase 5) is
  planned.