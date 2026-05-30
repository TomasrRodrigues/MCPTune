import pytest

from mcptune.schema.tools import ToolParameter, ToolSpec
from mcptune.synthesis.intent import IntentSynthesizer


def _make_tool(
    name: str = "get_weather",
    description: str = "Get the weather for a city",
    parameters: list[ToolParameter] | None = None,
) -> ToolSpec:
    if parameters is None:
        parameters = [
            ToolParameter(
                name="city",
                schema={"type": "string"},
                required=True,
                description="The city to query",
            )
        ]
    return ToolSpec(name=name, description=description, parameters=parameters)


# ---------------------------------------------------------------------------
# Template fallback
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_template_fallback_when_backend_none():
    synth = IntentSynthesizer(backend="none")
    result = synth.synthesize(_make_tool(), {"city": "Lisbon"})
    assert result.intent
    assert "Lisbon" in result.intent


@pytest.mark.unit
def test_template_fallback_includes_tool_name():
    synth = IntentSynthesizer(backend="none")
    result = synth.synthesize(_make_tool(name="search_docs"), {})
    result = synth.synthesize(_make_tool(name="search_docs"), {})
    assert "search_docs" in result.intent


@pytest.mark.unit
def test_template_fallback_handles_empty_arguments():
    synth = IntentSynthesizer(backend="none")
    result = synth.synthesize(_make_tool(name="ping", parameters=[]), {})
    assert result.intent


# ---------------------------------------------------------------------------
# LLM path — happy and degenerate responses
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_synthesize_returns_llm_text():
    synth = IntentSynthesizer(
        backend="ollama",
        llm_call=lambda _p: "What's the weather in Lisbon?",
    )
    result = synth.synthesize(_make_tool(), {"city": "Lisbon"})
    assert result.intent == "What's the weather in Lisbon?"


@pytest.mark.unit
def test_synthesize_strips_response_whitespace():
    synth = IntentSynthesizer(
        backend="ollama",
        llm_call=lambda _p: "  \n\nWhat's the weather?  \n  ",
    )
    result = synth.synthesize(_make_tool(), {"city": "Lisbon"})
    assert result.intent == "What's the weather?"


@pytest.mark.unit
def test_synthesize_falls_back_when_llm_raises():
    def raising(_p):
        raise RuntimeError("LLM unreachable")

    synth = IntentSynthesizer(backend="ollama", llm_call=raising)
    result = synth.synthesize(_make_tool(), {"city": "Lisbon"})
    assert "get_weather" in result.intent


@pytest.mark.unit
def test_synthesize_falls_back_when_llm_returns_empty():
    synth = IntentSynthesizer(backend="ollama", llm_call=lambda _p: "")
    result = synth.synthesize(_make_tool(), {"city": "Lisbon"})
    assert "get_weather" in result.intent


@pytest.mark.unit
def test_synthesize_falls_back_when_llm_returns_whitespace_only():
    synth = IntentSynthesizer(backend="ollama", llm_call=lambda _p: "   \n   ")
    result = synth.synthesize(_make_tool(), {"city": "Lisbon"})
    assert "get_weather" in result.intent


# ---------------------------------------------------------------------------
# Prompt content — what the LLM actually sees
# ---------------------------------------------------------------------------


def _capture_prompt():
    captured: dict = {}

    def call(prompt):
        captured["prompt"] = prompt
        return "test response"

    return captured, call


@pytest.mark.unit
def test_prompt_includes_tool_name_and_description():
    captured, call = _capture_prompt()
    synth = IntentSynthesizer(backend="ollama", llm_call=call)
    tool = _make_tool(name="get_weather", description="Get the weather for a city")
    synth.synthesize(tool, {})
    assert "get_weather" in captured["prompt"]
    assert "Get the weather for a city" in captured["prompt"]


@pytest.mark.unit
def test_prompt_includes_parameter_descriptions():
    captured, call = _capture_prompt()
    synth = IntentSynthesizer(backend="ollama", llm_call=call)
    tool = _make_tool(
        parameters=[
            ToolParameter(
                name="city",
                schema={"type": "string"},
                required=True,
                description="The city name to query",
            )
        ]
    )
    synth.synthesize(tool, {"city": "Lisbon"})
    assert "city" in captured["prompt"]
    assert "The city name to query" in captured["prompt"]


@pytest.mark.unit
def test_prompt_includes_argument_values():
    captured, call = _capture_prompt()
    synth = IntentSynthesizer(backend="ollama", llm_call=call)
    synth.synthesize(_make_tool(), {"city": "Lisbon", "units": "celsius"})
    assert "Lisbon" in captured["prompt"]
    assert "celsius" in captured["prompt"]


@pytest.mark.unit
def test_prompt_instructs_to_avoid_tool_name():
    captured, call = _capture_prompt()
    synth = IntentSynthesizer(backend="ollama", llm_call=call)
    synth.synthesize(_make_tool(), {})
    lower = captured["prompt"].lower()
    assert "tool name" in lower or "not mention" in lower


@pytest.mark.unit
def test_prompt_handles_missing_parameter_description():
    captured, call = _capture_prompt()
    synth = IntentSynthesizer(backend="ollama", llm_call=call)
    tool = _make_tool(
        parameters=[
            ToolParameter(name="city", schema={"type": "string"}, required=True, description="")
        ]
    )
    synth.synthesize(tool, {"city": "Lisbon"})
    assert "city" in captured["prompt"]


@pytest.mark.unit
def test_prompt_handles_no_parameters():
    captured, call = _capture_prompt()
    synth = IntentSynthesizer(backend="ollama", llm_call=call)
    synth.synthesize(_make_tool(parameters=[]), {})
    assert "(no parameters)" in captured["prompt"]


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_backend_is_none():
    assert IntentSynthesizer().backend == "none"


@pytest.mark.unit
def test_default_prompt_version():
    assert IntentSynthesizer().prompt_version == "intent_v1"


@pytest.mark.unit
def test_default_temperature_is_above_zero():
    """Intent synthesis wants variety, not determinism — default temp > 0."""
    assert IntentSynthesizer().temperature > 0


# ---------------------------------------------------------------------------
# Provenance — IntentResult.prompt_version
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_template_fallback_records_no_prompt_version():
    synth = IntentSynthesizer(backend="none")
    result = synth.synthesize(_make_tool(), {"city": "Lisbon"})
    assert result.prompt_version is None


@pytest.mark.unit
def test_llm_path_records_prompt_version():
    synth = IntentSynthesizer(
        backend="ollama",
        prompt_version="intent_v1",
        llm_call=lambda _p: "What's the weather in Lisbon?",
    )
    result = synth.synthesize(_make_tool(), {"city": "Lisbon"})
    assert result.prompt_version == "intent_v1"


@pytest.mark.unit
def test_llm_fallback_path_records_no_prompt_version():
    """When the LLM raises and we fall back to the template, the prompt
    version recorded must be None — the intent wasn't actually produced
    by the prompt."""

    def raising(_p):
        raise RuntimeError("LLM down")

    synth = IntentSynthesizer(backend="ollama", llm_call=raising)
    result = synth.synthesize(_make_tool(), {"city": "Lisbon"})
    assert result.prompt_version is None
