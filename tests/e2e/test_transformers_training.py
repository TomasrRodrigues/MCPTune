"""End-to-end smoke tests for TransformersTrainerBackend.

Marked @pytest.mark.e2e because they download a real (small) model and
run actual training steps. Opt-in: pytest -m e2e.

Skipped entirely when transformers/torch/peft/datasets aren't installed.
"""

import pytest

from mcptune.schema.dataset import DatasetRow

pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")
pytest.importorskip("datasets")

from mcptune.training.backends.transformers_backend import (  # noqa: E402
    TransformersTrainerBackend,
)
from mcptune.training.types import TrainedModel  # noqa: E402


TINY_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"


def _make_dataset(n: int = 4) -> list[DatasetRow]:
    return [
        DatasetRow(
            tool_name="get_weather",
            arguments={"city": "Lisbon"},
            request={
                "jsonrpc": "2.0",
                "id": f"req-{i}",
                "method": "tools/call",
                "params": {
                    "name": "get_weather",
                    "arguments": {"city": "Lisbon"},
                },
            },
            response={"temp": 22},
            user_intent="What's the weather in Lisbon?",
        )
        for i in range(n)
    ]


@pytest.mark.e2e
def test_transformers_backend_trains_a_real_model(tmp_path):
    backend = TransformersTrainerBackend(output_dir=str(tmp_path / "checkpoints"))

    result = backend.train(
        model_name=TINY_MODEL,
        dataset=_make_dataset(4),
        config={
            "epochs": 1,
            "max_length": 128,
            "lora_rank": 4,
            "lora_alpha": 8,
            "batch_size": 1,
        },
    )

    assert result.backend == "transformers"
    assert result.metadata["base_model"] == TINY_MODEL
    assert result.metadata["num_examples"] == 4
    assert result.metadata["lora_rank"] == 4


@pytest.mark.e2e
def test_transformers_backend_save_writes_adapter_to_disk(tmp_path):
    backend = TransformersTrainerBackend(output_dir=str(tmp_path / "checkpoints"))

    result = backend.train(
        model_name=TINY_MODEL,
        dataset=_make_dataset(4),
        config={"epochs": 1, "max_length": 128, "lora_rank": 4, "batch_size": 1},
    )

    save_dir = tmp_path / "final"
    backend.save(result, str(save_dir))

    assert save_dir.exists()
    assert (save_dir / "adapter_config.json").exists()
    assert result.model_path == str(save_dir)


@pytest.mark.e2e
def test_transformers_backend_raises_on_empty_dataset():
    backend = TransformersTrainerBackend()
    with pytest.raises(ValueError, match="empty dataset"):
        backend.train(model_name=TINY_MODEL, dataset=[])


@pytest.mark.e2e
def test_transformers_backend_save_before_train_raises(tmp_path):
    backend = TransformersTrainerBackend()
    with pytest.raises(RuntimeError, match="No trained model"):
        backend.save(TrainedModel(), str(tmp_path / "x"))