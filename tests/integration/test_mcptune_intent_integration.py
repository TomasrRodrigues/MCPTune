import pytest

from mcptune import MCPTune
from mcptune.schema import ToolParameter, ToolSpec


def _toy_tools():
    return [
        ToolSpec(
            name="get_weather",
            description="Get the weather for a city",
            parameters=[
                ToolParameter(
                    name="city",
                    schema={"type": "string"},
                    required=True,
                    description="The city to query",
                ),
            ],
        )
    ]


@pytest.mark.integration
def test_build_dataset_default_backend_uses_template_intent():
    """Default intent_backend='none' → every row has a template intent
    and intent_prompt_version=None."""
    tuner = MCPTune(model="test", mcpserver=None, seed=42)
    rows = tuner.build_dataset(_toy_tools(), samples_per_tool=2)

    for row in rows:
        assert row.user_intent
        assert "get_weather" in row.user_intent  # template signature
        assert row.intent_prompt_version is None


@pytest.mark.integration
def test_build_dataset_llm_backend_records_prompt_version():
    """When configured with an LLM backend (via llm_call injection), the
    rows record the synthesizer's prompt_version."""
    captured = []

    def fake_llm(prompt):
        captured.append(prompt)
        return "What's the weather in Porto today?"

    tuner = MCPTune(
        model="test",
        mcpserver=None,
        seed=42,
        intent_backend="ollama",
        intent_llm_call=fake_llm,
    )
    rows = tuner.build_dataset(_toy_tools(), samples_per_tool=2)

    assert len(captured) == 2  # one LLM call per row
    for row in rows:
        assert row.user_intent == "What's the weather in Porto today?"
        assert row.intent_prompt_version == "intent_v1"
