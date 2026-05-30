import json

import pytest

from mcptune.formats import (
    FORMATS,
    anthropic_tool_use,
    convert,
    openai_messages,
    sharegpt,
    trl_sft,
)
from mcptune.schema.dataset import DatasetRow


@pytest.fixture
def sample_row():
    return DatasetRow(
        tool_name="get_weather",
        arguments={"city": "Lisbon"},
        request={
            "jsonrpc": "2.0",
            "id": "abc-123",
            "method": "tools/call",
            "params": {"name": "get_weather", "arguments": {"city": "Lisbon"}},
        },
        response={"temp": 22, "conditions": "sunny"},
        user_intent="What's the weather in Lisbon?",
        intent_prompt_version="intent_v1",
    )


@pytest.fixture
def sample_row_no_response():
    """A row that wasn't executed (response is None)."""
    return DatasetRow(
        tool_name="ping",
        arguments={},
        request={
            "jsonrpc": "2.0",
            "id": "def-456",
            "method": "tools/call",
            "params": {"name": "ping", "arguments": {}},
        },
        user_intent="Are you there?",
    )


# ---------------------------------------------------------------------------
# Registry / convert
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_registry_has_four_formats():
    assert set(FORMATS) == {"openai", "sharegpt", "trl", "anthropic"}


@pytest.mark.unit
def test_convert_dispatches_to_registered_format(sample_row):
    assert convert([sample_row], "openai") == openai_messages([sample_row])


@pytest.mark.unit
def test_convert_unknown_format_raises_value_error():
    with pytest.raises(ValueError, match="Unknown format"):
        convert([], "made_up_format")


@pytest.mark.unit
def test_convert_unknown_format_lists_available():
    with pytest.raises(ValueError, match="openai"):
        convert([], "made_up_format")


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_openai_emits_one_conversation_per_row(sample_row):
    assert len(openai_messages([sample_row, sample_row])) == 2


@pytest.mark.unit
def test_openai_three_messages_user_assistant_tool(sample_row):
    conv = openai_messages([sample_row])[0]
    assert [m["role"] for m in conv["messages"]] == ["user", "assistant", "tool"]


@pytest.mark.unit
def test_openai_user_message_uses_user_intent(sample_row):
    conv = openai_messages([sample_row])[0]
    assert conv["messages"][0]["content"] == "What's the weather in Lisbon?"


@pytest.mark.unit
def test_openai_assistant_tool_call_uses_row_request_id(sample_row):
    conv = openai_messages([sample_row])[0]
    assert conv["messages"][1]["tool_calls"][0]["id"] == "abc-123"


@pytest.mark.unit
def test_openai_assistant_arguments_are_json_string(sample_row):
    conv = openai_messages([sample_row])[0]
    args = conv["messages"][1]["tool_calls"][0]["function"]["arguments"]
    assert isinstance(args, str)
    assert json.loads(args) == {"city": "Lisbon"}


@pytest.mark.unit
def test_openai_tool_message_carries_response(sample_row):
    conv = openai_messages([sample_row])[0]
    tool_msg = conv["messages"][2]
    assert tool_msg["tool_call_id"] == "abc-123"
    assert json.loads(tool_msg["content"]) == {"temp": 22, "conditions": "sunny"}


@pytest.mark.unit
def test_openai_empty_response_becomes_empty_string(sample_row_no_response):
    conv = openai_messages([sample_row_no_response])[0]
    assert conv["messages"][2]["content"] == ""


@pytest.mark.unit
def test_openai_fallback_when_user_intent_missing():
    row = DatasetRow(
        tool_name="ping",
        arguments={},
        request={"jsonrpc": "2.0", "id": "x", "method": "tools/call", "params": {}},
    )
    conv = openai_messages([row])[0]
    assert "ping" in conv["messages"][0]["content"]


# ---------------------------------------------------------------------------
# ShareGPT
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sharegpt_uses_conversations_field(sample_row):
    assert "conversations" in sharegpt([sample_row])[0]


@pytest.mark.unit
def test_sharegpt_uses_from_value_keys(sample_row):
    for turn in sharegpt([sample_row])[0]["conversations"]:
        assert set(turn.keys()) == {"from", "value"}


@pytest.mark.unit
def test_sharegpt_role_order_human_gpt_tool(sample_row):
    convs = sharegpt([sample_row])[0]["conversations"]
    assert [t["from"] for t in convs] == ["human", "gpt", "tool"]


@pytest.mark.unit
def test_sharegpt_gpt_turn_contains_tool_call(sample_row):
    gpt_value = json.loads(sharegpt([sample_row])[0]["conversations"][1]["value"])
    assert gpt_value["tool_call"]["name"] == "get_weather"
    assert gpt_value["tool_call"]["arguments"] == {"city": "Lisbon"}


# ---------------------------------------------------------------------------
# TRL SFT
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_trl_uses_messages_field(sample_row):
    assert "messages" in trl_sft([sample_row])[0]


@pytest.mark.unit
def test_trl_role_order(sample_row):
    msgs = trl_sft([sample_row])[0]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "tool"]


@pytest.mark.unit
def test_trl_all_content_is_string(sample_row):
    """TRL stringifies everything for chat templates that don't natively
    handle structured tool calls."""
    for m in trl_sft([sample_row])[0]["messages"]:
        assert isinstance(m["content"], str)


# ---------------------------------------------------------------------------
# Anthropic tool_use
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_anthropic_uses_messages_field(sample_row):
    assert "messages" in anthropic_tool_use([sample_row])[0]


@pytest.mark.unit
def test_anthropic_role_order_user_assistant_user(sample_row):
    """Anthropic returns tool_result on a user turn, not a separate role."""
    msgs = anthropic_tool_use([sample_row])[0]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]


@pytest.mark.unit
def test_anthropic_assistant_uses_tool_use_block(sample_row):
    block = anthropic_tool_use([sample_row])[0]["messages"][1]["content"][0]
    assert block["type"] == "tool_use"
    assert block["id"] == "abc-123"
    assert block["name"] == "get_weather"
    assert block["input"] == {"city": "Lisbon"}


@pytest.mark.unit
def test_anthropic_tool_result_block_references_call_id(sample_row):
    block = anthropic_tool_use([sample_row])[0]["messages"][2]["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "abc-123"
