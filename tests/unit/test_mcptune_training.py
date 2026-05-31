import pytest

from mcptune.mcptune import MCPTune
from mcptune.training.backends.mock_backend import MockTrainerBackend


@pytest.mark.unit
def test_mcptune_delegates_training_to_backend():
    trainer = MockTrainerBackend()
    mcp = MCPTune(model="test-model", mcpserver=None, trainer=trainer)
    result = mcp.train([])
    assert result.backend == "mock"
