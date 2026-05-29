import os

import httpx
import pytest

from mcptune.sampling.semantic import SemanticSampler


def _ollama_running() -> bool:
    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    try:
        httpx.get(f"{host}/api/tags", timeout=2.0).raise_for_status()
        return True
    except Exception:
        return False


@pytest.mark.e2e
@pytest.mark.skipif(not _ollama_running(), reason="Ollama is not running")
def test_ollama_real_call():
    sampler = SemanticSampler(backend="ollama")
    result = sampler.sample_batch("weather", "Get weather", {"city": {"type": "string"}})
    assert "city" in result
    assert isinstance(result["city"], str)
