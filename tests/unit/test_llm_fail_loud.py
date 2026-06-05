# tests/unit/test_llm_fail_loud.py
import pytest

from mcptune.schema.tools import ToolParameter, ToolSpec
from mcptune.synthesis.intent import IntentSynthesizer


def _tool():
    return ToolSpec(
        "get_weather", "Get weather", [ToolParameter("city", {"type": "string"}, True, "")]
    )


def test_strict_raises_on_backend_failure():
    def boom(_):
        raise ConnectionError("ollama down")

    syn = IntentSynthesizer(backend="ollama", llm_call=boom, strict=True)
    with pytest.raises(RuntimeError):
        syn.synthesize(_tool(), {"city": "Lisbon"})


def test_non_strict_warns_once_and_counts():
    def boom(_):
        raise ConnectionError("down")

    syn = IntentSynthesizer(backend="ollama", llm_call=boom, strict=False)
    for _ in range(5):
        syn.synthesize(_tool(), {"city": "Lisbon"})
    assert syn.fallback_count == 5
    assert syn._warned is True


def test_none_backend_never_counts_as_fallback():
    syn = IntentSynthesizer(backend="none")
    syn.synthesize(_tool(), {"city": "Lisbon"})
    assert syn.fallback_count == 0
