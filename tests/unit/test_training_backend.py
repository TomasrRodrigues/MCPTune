import pytest

from mcptune.training.backends.transformers_backend import (
    TransformersTrainerBackend,
)
from mcptune.training.types import TrainedModel


@pytest.mark.unit
def test_transformers_backend_returns_trained_model():
    backend = TransformersTrainerBackend()

    model = backend.train(
        model_name="test-model",
        dataset=[],
        config={},
    )

    assert isinstance(model, TrainedModel)


@pytest.mark.unit
def test_transformers_backend_sets_backend_name():
    backend = TransformersTrainerBackend()

    model = backend.train(
        model_name="test-model",
        dataset=[],
        config={},
    )

    assert model.backend == "transformers"


@pytest.mark.unit
def test_transformers_backend_records_model_name():
    backend = TransformersTrainerBackend()

    model = backend.train(
        model_name="tiny-model",
        dataset=[1, 2, 3],
        config={},
    )

    assert model.metadata["model_name"] == "tiny-model"


@pytest.mark.unit
def test_transformers_backend_records_dataset_size():
    backend = TransformersTrainerBackend()

    dataset = [1, 2, 3, 4]

    model = backend.train(
        model_name="tiny-model",
        dataset=dataset,
        config={},
    )

    assert model.metadata["num_examples"] == len(dataset)


@pytest.mark.unit
def test_save_updates_model_path():
    backend = TransformersTrainerBackend()

    model = backend.train(
        model_name="tiny-model",
        dataset=[],
        config={},
    )

    backend.save(model, "artifacts/model")

    assert model.model_path == "artifacts/model"
