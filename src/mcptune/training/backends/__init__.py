"""Concrete TrainerBackend implementations.

- TransformersTrainerBackend: real LoRA fine-tuning via transformers +
  PEFT. Requires mcptune[transformers].
- MockTrainerBackend: no-op backend for testing and dry runs.
"""

from .mock_backend import MockTrainerBackend
from .transformers_backend import TransformersTrainerBackend

__all__ = ["MockTrainerBackend", "TransformersTrainerBackend"]