"""Run a model against an in-process MCP server through the runtime.

Use a real instruct model (Qwen2.5-1.5B-Instruct or larger) to see calls
actually happen — 135M won't reliably emit them. Requires
mcptune[transformers].
"""

import asyncio

from fastmcp import FastMCP
from transformers import AutoModelForCausalLM, AutoTokenizer

from mcptune.adapters.fastmcp import FastMCPAdapter
from mcptune.runtime import TransformersModelRunner, run

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def build_server() -> FastMCP:
    server = FastMCP("runtime-demo")

    @server.tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Sunny and 22C in {city}"

    return server


async def main() -> None:
    server = build_server()
    adapter = FastMCPAdapter(server)
    tools = await adapter.discover_tools()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL)
    runner = TransformersModelRunner(model, tokenizer)

    result = await run(runner, adapter, "What's the weather in Lisbon?", tools)

    print("--- trajectory ---")
    for m in result.messages:
        print(f"[{m['role']}] {m['content']}")
    print("\n--- final ---")
    print(result.final_text)
    print(f"\ntool calls: {[(c.name, c.arguments) for c in result.tool_calls]}")
    print(f"stopped: {result.stopped_reason}")


if __name__ == "__main__":
    asyncio.run(main())