import pytest

from mcptune.llm.client import LLMClient


@pytest.mark.unit
def test_llm_client_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        LLMClient(backend="local")  # not an LLM backend


@pytest.mark.unit
def test_llm_client_rejects_none_backend():
    with pytest.raises(ValueError, match="Unknown LLM backend"):
        LLMClient(backend="none")


@pytest.mark.unit
def test_llm_client_accepts_ollama():
    client = LLMClient(backend="ollama")
    assert client.backend == "ollama"


@pytest.mark.unit
def test_llm_client_accepts_transformers():
    client = LLMClient(backend="transformers")
    assert client.backend == "transformers"


@pytest.mark.unit
def test_llm_client_ollama_host_falls_back_to_env():
    import os

    os.environ["OLLAMA_HOST"] = "http://custom-host:9999"
    try:
        client = LLMClient(backend="ollama")
        assert client.ollama_host == "http://custom-host:9999"
    finally:
        del os.environ["OLLAMA_HOST"]


@pytest.mark.unit
def test_llm_client_explicit_host_overrides_env():
    import os

    os.environ["OLLAMA_HOST"] = "http://env-host:1111"
    try:
        client = LLMClient(backend="ollama", ollama_host="http://explicit:2222")
        assert client.ollama_host == "http://explicit:2222"
    finally:
        del os.environ["OLLAMA_HOST"]


@pytest.mark.unit
def test_llm_client_ollama_unreachable_raises_connection_error():
    client = LLMClient(backend="ollama", ollama_host="http://localhost:1")
    with pytest.raises(ConnectionError, match="Ollama"):
        client.generate("test prompt")
