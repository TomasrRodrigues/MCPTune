import pytest

from mcptune.mcptune import MCPTune
from mcptune.training.backends.transformers_backend import (
    TransformersTrainerBackend,
)


@pytest.mark.unit
def test_mcptune_delegates_training_to_backend():
    trainer = TransformersTrainerBackend()

    mcp = MCPTune(
        model="test-model",
        mcpserver=None,
        trainer=trainer,
    )

    result = mcp.train([])

    assert result.backend == "transformers"
