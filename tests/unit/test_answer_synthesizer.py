from mcptune.schema.tools import ToolParameter, ToolSpec
from mcptune.synthesis.answer import AnswerResult, AnswerSynthesizer


def _tool():
    return ToolSpec(
        "get_weather", "Get the weather.", [ToolParameter("city", {"type": "string"}, True, "")]
    )


def test_none_backend_uses_template_and_null_provenance():
    out = AnswerSynthesizer(backend="none").synthesize(_tool(), {"city": "Lisbon"}, "Sunny, 22C")
    assert isinstance(out, AnswerResult)
    assert out.prompt_version is None
    assert "Sunny, 22C" in out.answer


def test_injected_llm_is_used_and_versioned():
    out = AnswerSynthesizer(
        backend="ollama", llm_call=lambda p: "It's sunny and 22°C in Lisbon."
    ).synthesize(_tool(), {"city": "Lisbon"}, "Sunny, 22C")
    assert out.answer == "It's sunny and 22°C in Lisbon."
    assert out.prompt_version == "answer_v1"


def test_llm_failure_falls_back_to_template():
    def boom(_):
        raise RuntimeError("down")

    out = AnswerSynthesizer(backend="ollama", llm_call=boom).synthesize(_tool(), {}, "ok")
    assert out.prompt_version is None
