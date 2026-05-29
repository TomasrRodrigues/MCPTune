import re
import uuid
from unittest.mock import patch

import pytest

from mcptune.sampling.lookups import LookupRule, lookup_value
from mcptune.sampling.semantic import SemanticSampler

# ---------------------------------------------------------------------------
# Rule engine (lookup_value)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_lookup_format_email_returns_valid_email():
    value = lookup_value("contact", {"type": "string", "format": "email"})
    assert isinstance(value, str)
    assert "@" in value and "." in value


@pytest.mark.unit
def test_lookup_format_uri_returns_url():
    value = lookup_value("link", {"type": "string", "format": "uri"})
    assert isinstance(value, str)
    assert value.startswith(("http://", "https://"))


@pytest.mark.unit
def test_lookup_format_date_returns_iso_date():
    value = lookup_value("when", {"type": "string", "format": "date"})
    assert re.match(r"^\d{4}-\d{2}-\d{2}$", value)


@pytest.mark.unit
def test_lookup_format_datetime_returns_iso_datetime():
    value = lookup_value("ts", {"type": "string", "format": "date-time"})
    assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value)


@pytest.mark.unit
def test_lookup_format_uuid_returns_valid_uuid():
    value = lookup_value("id", {"type": "string", "format": "uuid"})
    uuid.UUID(value)  # raises ValueError if invalid


@pytest.mark.unit
def test_lookup_name_city_returns_string():
    value = lookup_value("city", {"type": "string"})
    assert isinstance(value, str) and len(value) > 0


@pytest.mark.unit
def test_lookup_name_country_returns_string():
    value = lookup_value("country", {"type": "string"})
    assert isinstance(value, str) and len(value) > 0


@pytest.mark.unit
def test_lookup_examples_field_takes_precedence_over_rules():
    value = lookup_value(
        "city",
        {"type": "string", "examples": ["Tokyo", "Osaka"]},
    )
    assert value == "Tokyo"


@pytest.mark.unit
def test_lookup_unknown_parameter_returns_none():
    value = lookup_value("xyz_arbitrary_thing", {"type": "string"})
    assert value is None


@pytest.mark.unit
def test_lookup_custom_rule_overrides_default():
    custom = (LookupRule(("custom",), (), "custom-value"),)
    value = lookup_value("custom_param", {"type": "string"}, rules=custom)
    assert value == "custom-value"


@pytest.mark.unit
def test_lookup_format_match_beats_name_match():
    # name contains "city" (would map to Lisbon), but format is "email"
    value = lookup_value("city_email", {"type": "string", "format": "email"})
    assert "@" in value


# ---------------------------------------------------------------------------
# SemanticSampler.sample_batch
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sample_batch_local_backend_returns_known_params():
    sampler = SemanticSampler(backend="local")
    properties = {
        "city": {"type": "string"},
        "email": {"type": "string", "format": "email"},
    }
    result = sampler.sample_batch("tool", "desc", properties)
    assert set(result.keys()) == {"city", "email"}
    assert "@" in result["email"]


@pytest.mark.unit
def test_sample_batch_omits_unknown_params():
    sampler = SemanticSampler(backend="local")
    properties = {
        "city": {"type": "string"},
        "xyz_arbitrary": {"type": "string"},
    }
    result = sampler.sample_batch("tool", "desc", properties)
    assert "city" in result
    assert "xyz_arbitrary" not in result


@pytest.mark.unit
def test_sample_batch_none_backend_returns_empty():
    sampler = SemanticSampler(backend="none")
    result = sampler.sample_batch("tool", "desc", {"city": {"type": "string"}})
    assert result == {}


@pytest.mark.unit
def test_sample_batch_empty_properties_returns_empty():
    sampler = SemanticSampler(backend="local")
    assert sampler.sample_batch("tool", "desc", {}) == {}


@pytest.mark.unit
def test_sample_batch_uses_cache_on_repeat_calls():
    sampler = SemanticSampler(backend="local")
    properties = {"city": {"type": "string"}}

    with patch.object(sampler, "_execute_local_lookup", wraps=sampler._execute_local_lookup) as spy:
        sampler.sample_batch("tool", "desc", properties)
        sampler.sample_batch("tool", "desc", properties)
        assert spy.call_count == 1, "second call should hit the cache"


@pytest.mark.unit
def test_sample_batch_different_tools_have_independent_cache_entries():
    sampler = SemanticSampler(backend="local")
    properties = {"city": {"type": "string"}}

    sampler.sample_batch("tool_a", "desc_a", properties)
    sampler.sample_batch("tool_b", "desc_b", properties)

    # Two distinct cache entries → two separate keys
    assert len(sampler.cache._storage) == 2


# ---------------------------------------------------------------------------
# LLM-backed sampling — injected llm_call for hermetic tests
# ---------------------------------------------------------------------------


def fake_llm(response_text: str):
    """Build an llm_call that ignores its input and returns a fixed response."""
    return lambda prompt: response_text


@pytest.mark.unit
def test_llm_returns_clean_json_object():
    sampler = SemanticSampler(
        backend="ollama",
        llm_call=fake_llm('{"city": "Tokyo", "country": "Japan"}'),
    )
    result = sampler.sample_batch(
        "weather",
        "Get weather",
        {"city": {"type": "string"}, "country": {"type": "string"}},
    )
    assert result == {"city": "Tokyo", "country": "Japan"}


@pytest.mark.unit
def test_llm_response_with_markdown_fence_still_parses():
    sampler = SemanticSampler(
        backend="ollama",
        llm_call=fake_llm('Here you go:\n```json\n{"city": "Osaka"}\n```'),
    )
    result = sampler.sample_batch("weather", "Get weather", {"city": {"type": "string"}})
    assert result == {"city": "Osaka"}


@pytest.mark.unit
def test_llm_response_with_prose_around_json_still_parses():
    sampler = SemanticSampler(
        backend="ollama",
        llm_call=fake_llm('The answer is {"city": "Madrid"} and that is all.'),
    )
    result = sampler.sample_batch("weather", "Get weather", {"city": {"type": "string"}})
    assert result == {"city": "Madrid"}


@pytest.mark.unit
def test_llm_garbage_response_falls_back_to_local():
    sampler = SemanticSampler(
        backend="ollama",
        llm_call=fake_llm("I don't know what to put here."),
    )
    result = sampler.sample_batch("weather", "Get weather", {"city": {"type": "string"}})
    assert "city" in result


@pytest.mark.unit
def test_llm_value_with_wrong_type_is_dropped():
    sampler = SemanticSampler(
        backend="ollama",
        llm_call=fake_llm('{"city": "Lisbon", "age": "not_an_integer"}'),
    )
    result = sampler.sample_batch(
        "register",
        "Register a user",
        {"city": {"type": "string"}, "age": {"type": "integer"}},
    )
    assert result == {"city": "Lisbon"}


@pytest.mark.unit
def test_llm_hallucinated_parameter_is_dropped():
    sampler = SemanticSampler(
        backend="ollama",
        llm_call=fake_llm('{"city": "Lisbon", "extra": "junk"}'),
    )
    result = sampler.sample_batch("weather", "Get weather", {"city": {"type": "string"}})
    assert result == {"city": "Lisbon"}


@pytest.mark.unit
def test_llm_call_raising_falls_back_to_local():
    def raising_call(_prompt):
        raise RuntimeError("network is on fire")

    sampler = SemanticSampler(backend="ollama", llm_call=raising_call)
    result = sampler.sample_batch("weather", "Get weather", {"city": {"type": "string"}})
    assert "city" in result


@pytest.mark.unit
def test_prompt_carries_tool_metadata_and_properties():
    captured = {}

    def capturing_call(prompt):
        captured["prompt"] = prompt
        return '{"city": "Lisbon"}'

    sampler = SemanticSampler(backend="ollama", llm_call=capturing_call)
    sampler.sample_batch(
        "get_weather",
        "Get the weather for a city",
        {"city": {"type": "string", "description": "city name"}},
    )

    assert "get_weather" in captured["prompt"]
    assert "Get the weather for a city" in captured["prompt"]
    assert "city" in captured["prompt"]


@pytest.mark.unit
def test_extract_json_handles_direct_object():
    assert SemanticSampler._extract_json('{"a": 1}') == {"a": 1}


@pytest.mark.unit
def test_extract_json_handles_non_dict_top_level():
    assert SemanticSampler._extract_json("[1, 2, 3]") == {}


@pytest.mark.unit
def test_extract_json_handles_empty_string():
    assert SemanticSampler._extract_json("") == {}


@pytest.mark.unit
def test_ollama_unreachable_raises_helpful_error():
    """When Ollama isn't running and no llm_call is injected, the user
    should get a clear connection error."""
    sampler = SemanticSampler(
        backend="ollama",
        ollama_host="http://localhost:1",  # nothing listening here
    )
    # The connection error is caught in sample_batch and falls back, so
    # we exercise _call_ollama directly to see the raised exception.
    with pytest.raises(ConnectionError, match="Ollama"):
        sampler._call_ollama("test prompt")


# ---------------------------------------------------------------------------
# Description grounding
# ---------------------------------------------------------------------------


def _capture_prompt():
    """Build an llm_call that records the prompt and returns a valid response."""
    captured = {}

    def call(prompt):
        captured["prompt"] = prompt
        return '{"city": "Lisbon"}'

    return captured, call


@pytest.mark.unit
def test_grounded_prompt_includes_tool_description():
    captured, call = _capture_prompt()
    sampler = SemanticSampler(backend="ollama", llm_call=call)
    sampler.sample_batch(
        "get_weather",
        "Get the current weather for a city",
        {"city": {"type": "string"}},
    )
    assert "Get the current weather for a city" in captured["prompt"]


@pytest.mark.unit
def test_grounded_prompt_includes_parameter_description():
    captured, call = _capture_prompt()
    sampler = SemanticSampler(backend="ollama", llm_call=call)
    sampler.sample_batch(
        "get_weather",
        "Get the weather",
        {"city": {"type": "string", "description": "The city to query"}},
    )
    assert "The city to query" in captured["prompt"]


@pytest.mark.unit
def test_grounded_prompt_handles_missing_parameter_description():
    captured, call = _capture_prompt()
    sampler = SemanticSampler(backend="ollama", llm_call=call)
    sampler.sample_batch(
        "get_weather",
        "Get the weather",
        {"city": {"type": "string"}},  # no description
    )
    # Prompt should still include the parameter name and type
    assert "city" in captured["prompt"]
    assert "string" in captured["prompt"]


@pytest.mark.unit
def test_grounded_prompt_handles_empty_tool_description():
    captured, call = _capture_prompt()
    sampler = SemanticSampler(backend="ollama", llm_call=call)
    sampler.sample_batch("get_weather", "", {"city": {"type": "string"}})
    assert "(no description)" in captured["prompt"]


@pytest.mark.unit
def test_grounded_prompt_handles_none_description():
    """Schemas may store None where you'd expect an empty string."""
    captured, call = _capture_prompt()
    sampler = SemanticSampler(backend="ollama", llm_call=call)
    sampler.sample_batch(
        "get_weather",
        "Get the weather",
        {"city": {"type": "string", "description": None}},
    )
    assert "city" in captured["prompt"]


@pytest.mark.unit
def test_grounded_default_prompt_version():
    sampler = SemanticSampler(backend="local")
    assert sampler.prompt_version == "grounded_semantic_v1"


@pytest.mark.unit
def test_legacy_prompt_version_still_usable():
    """semantic_v1 stays available for ablation and dataset reproducibility."""
    captured, call = _capture_prompt()
    sampler = SemanticSampler(
        backend="ollama",
        prompt_version="semantic_v1",
        llm_call=call,
    )
    sampler.sample_batch(
        "get_weather",
        "Get the weather",
        {"city": {"type": "string", "description": "The city"}},
    )
    # The ungrounded prompt embeds the description in JSON, not in a
    # human-readable block — but the description is still in the prompt.
    assert "city" in captured["prompt"]


@pytest.mark.unit
def test_format_parameter_block_with_description():
    block = SemanticSampler._format_parameter_block(
        {"city": {"type": "string", "description": "The city name"}}
    )
    assert block == "- city (string): The city name"


@pytest.mark.unit
def test_format_parameter_block_with_format():
    block = SemanticSampler._format_parameter_block(
        {"contact": {"type": "string", "format": "email", "description": "Contact email"}}
    )
    assert "format: email" in block
    assert "Contact email" in block


@pytest.mark.unit
def test_format_parameter_block_without_description():
    block = SemanticSampler._format_parameter_block({"city": {"type": "string"}})
    assert block == "- city (string)"


@pytest.mark.unit
def test_truncate_short_text_unchanged():
    assert SemanticSampler._truncate("hello", 800) == "hello"


@pytest.mark.unit
def test_truncate_long_text_clipped_with_ellipsis():
    truncated = SemanticSampler._truncate("a" * 1000, 800)
    assert len(truncated) == 800
    assert truncated.endswith("...")
