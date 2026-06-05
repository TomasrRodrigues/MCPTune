"""Verify SFT label masking: prompt masked, assistant completion supervised."""

import pytest

from mcptune.formats.tool_use import ToolUseFormat
from mcptune.schema.dataset import DatasetRow
from mcptune.schema.tools import ToolParameter, ToolSpec

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@pytest.mark.integration
def test_prompt_masked_completion_supervised():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:
        pytest.skip(f"Qwen tokenizer unavailable: {e}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from mcptune.training.backends.transformers_backend import _tokenize_example

    tool = ToolSpec(
        "get_weather",
        "Get the weather for a city.",
        [ToolParameter("city", {"type": "string"}, True, "")],
        raw_input_schema={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    row = DatasetRow(
        tool_name="get_weather",
        arguments={"city": "Lisbon"},
        request={"id": "x"},
        user_intent="What's the weather in Lisbon?",
    )
    ex = ToolUseFormat().format_tool([row], [tool])[0]

    out = _tokenize_example(tokenizer, ex, max_length=256)
    labels, input_ids = out["labels"], out["input_ids"]

    # Some prompt tokens are masked, some completion tokens are supervised.
    assert any(label == -100 for label in labels)
    assert any(label != -100 for label in labels)

    supervised = [tok for tok, lab in zip(input_ids, labels, strict=True) if lab != -100]
    decoded = tokenizer.decode(supervised)
    assert "get_weather" in decoded and "<tool_call>" in decoded

    for mask, lab in zip(out["attention_mask"], labels, strict=True):
        if mask == 0:
            assert lab == -100


@pytest.mark.integration
def test_both_assistant_turns_supervised_in_full_loop():
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL)
    except Exception as e:
        pytest.skip(f"Qwen tokenizer unavailable: {e}")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from mcptune.training.backends.transformers_backend import _tokenize_example

    ex = {
        "messages": [
            {"role": "user", "content": "What's the weather in Lisbon?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": {"city": "Lisbon"}},
                    }
                ],
            },
            {"role": "tool", "name": "get_weather", "content": "Sunny and 22C in Lisbon"},
            {"role": "assistant", "content": "It's sunny and 22°C in Lisbon right now."},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                },
            }
        ],
    }

    out = _tokenize_example(tokenizer, ex, max_length=256)
    supervised = tokenizer.decode(
        [t for t, lab in zip(out["input_ids"], out["labels"], strict=True) if lab != -100]
    )
    # Both assistant turns are in the supervised region; the user turn is not.
    assert "<tool_call>" in supervised and "get_weather" in supervised
    assert "sunny" in supervised.lower()
    assert "What's the weather" not in supervised
