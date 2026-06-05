"""Minimal before/after evaluation - proof the fine-tune teaches tool calls.

Runs held-out intents through the runtime (mcptune.runtime) and scores
the emitted call against an expected one. This is the MINIMAL version:
tool-name + argument-key match. Splits, value-level metrics, per-tool
reports, and CI gating are the full pipeline (0.2.0).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..runtime import parse_tool_calls
from ..runtime.runner import ModelRunner
from ..schema.function_schema import toolspecs_to_function_schemas
from ..schema.tools import ToolSpec


@dataclass
class EvalCase:
    intent: str
    expected_tool: str
    expected_arg_keys: set[str]


@dataclass
class EvalReport:
    total: int
    tool_correct: int  # emitted a call to the right tool
    args_correct: int  # right tool AND argument keys match

    @property
    def tool_accuracy(self) -> float:
        return self.tool_correct / self.total if self.total else 0.0

    @property
    def arg_accuracy(self) -> float:
        return self.args_correct / self.total if self.total else 0.0


def evaluate(
    runner: ModelRunner,
    cases: list[EvalCase],
    tools: list[ToolSpec],
    *,
    call_format: str = "qwen",
) -> EvalReport:
    """Score one model (base or tuned) on held-out intents.

    Generates a single turn per case with the tools in context, parses
    the emitted call, and checks tool name + argument keys. No execution
    - we're measuring whether the model EMITS the right call.
    """
    function_schemas = toolspecs_to_function_schemas(tools)
    tool_ok = 0
    args_ok = 0

    for case in cases:
        messages = [{"role": "user", "content": case.intent}]
        text = runner.generate(messages, function_schemas)
        calls = parse_tool_calls(text, fmt=call_format)

        match = next((c for c in calls if c.name == case.expected_tool), None)
        if match is None:
            continue
        tool_ok += 1
        if set(match.arguments.keys()) == case.expected_arg_keys:
            args_ok += 1

    return EvalReport(total=len(cases), tool_correct=tool_ok, args_correct=args_ok)


def format_comparison(base: EvalReport, tuned: EvalReport) -> str:
    return (
        "Tool-call accuracy on held-out intents:\n"
        f"  tool name : base {base.tool_correct}/{base.total} "
        f"({base.tool_accuracy:.0%})  ->  tuned {tuned.tool_correct}/{tuned.total} "
        f"({tuned.tool_accuracy:.0%})\n"
        f"  + arg keys: base {base.args_correct}/{base.total} "
        f"({base.arg_accuracy:.0%})  ->  tuned {tuned.args_correct}/{tuned.total} "
        f"({tuned.arg_accuracy:.0%})"
    )
