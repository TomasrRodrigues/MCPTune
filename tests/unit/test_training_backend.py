"""Unit tests for the TrainerBackend abstraction layer.

These tests use MockTrainerBackend to exercise the contract without
depending on transformers/torch/peft. End-to-end tests against the
real TransformersTrainerBackend live in test_transformers_training.py
(marked @pytest.mark.e2e).
"""

import pytest

from mcptune.training.backends.mock_backend import MockTrainerBackend
from mcptune.training.types import TrainedModel


@pytest.mark.unit
def test_mock_backend_returns_trained_model():
    backend = MockTrainerBackend()
    model = backend.train(model_name="test-model", dataset=[], config={})
    assert isinstance(model, TrainedModel)


@pytest.mark.unit
def test_mock_backend_sets_backend_name():
    backend = MockTrainerBackend()
    model = backend.train(model_name="test-model", dataset=[], config={})
    assert model.backend == "mock"


@pytest.mark.unit
def test_mock_backend_records_model_name():
    backend = MockTrainerBackend()
    model = backend.train(model_name="tiny-model", dataset=[1, 2, 3], config={})
    assert model.metadata["model_name"] == "tiny-model"


@pytest.mark.unit
def test_mock_backend_records_dataset_size():
    backend = MockTrainerBackend()
    dataset = [1, 2, 3, 4]
    model = backend.train(model_name="tiny-model", dataset=dataset, config={})
    assert model.metadata["num_examples"] == len(dataset)


@pytest.mark.unit
def test_save_updates_model_path(tmp_path):
    backend = MockTrainerBackend()
    model = backend.train(model_name="tiny-model", dataset=[], config={})
    backend.save(model, str(tmp_path / "model"))
    assert model.model_path == str(tmp_path / "model")
