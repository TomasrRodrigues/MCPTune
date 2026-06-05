from mcptune.evaluation import EvalCase, evaluate, format_comparison
from mcptune.schema.tools import ToolParameter, ToolSpec


class ScriptedRunner:
    def __init__(self, outputs):
        self.outputs, self.i = list(outputs), 0

    def generate(self, messages, tools):
        out = self.outputs[self.i]
        self.i += 1
        return out


def _tools():
    return [
        ToolSpec(
            "get_weather",
            "Get weather",
            [ToolParameter("city", {"type": "string"}, True, "")],
            raw_input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    ]


def _cases():
    return [
        EvalCase("weather in Lisbon?", "get_weather", {"city"}),
        EvalCase("weather in Porto?", "get_weather", {"city"}),
    ]


def test_scores_correct_call():
    runner = ScriptedRunner(
        [
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Lisbon"}}</tool_call>',
            '<tool_call>{"name": "get_weather", "arguments": {"city": "Porto"}}</tool_call>',
        ]
    )
    rep = evaluate(runner, _cases(), _tools())
    assert rep.tool_correct == 2 and rep.args_correct == 2


def test_wrong_tool_and_missing_args_dont_count():
    runner = ScriptedRunner(
        [
            "It's sunny in Lisbon.",  # no call at all
            '<tool_call>{"name": "get_weather", "arguments": {}}</tool_call>',  # right tool, wrong keys
        ]
    )
    rep = evaluate(runner, _cases(), _tools())
    assert rep.tool_correct == 1 and rep.args_correct == 0


def test_comparison_string_renders():
    base = evaluate(ScriptedRunner(["x", "y"]), _cases(), _tools())
    tuned = evaluate(
        ScriptedRunner(
            [
                '<tool_call>{"name": "get_weather", "arguments": {"city": "Lisbon"}}</tool_call>',
                '<tool_call>{"name": "get_weather", "arguments": {"city": "Porto"}}</tool_call>',
            ]
        ),
        _cases(),
        _tools(),
    )
    out = format_comparison(base, tuned)
    assert "base 0/2" in out and "tuned 2/2" in out
