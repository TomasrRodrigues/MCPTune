"""Minimal end-to-end MCPTune example.

Runs against an in-process FastMCP server. No external services
required - uses the default local sampler and template-based intent
synthesis.

Usage:
    pip install mcptune
    python examples/quickstart.py
"""

import asyncio

from fastmcp import FastMCP

from mcptune import MCPTune


def build_server() -> FastMCP:
    server = FastMCP("quickstart-demo")

    @server.tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Sunny and 22C in {city}"

    @server.tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    return server


async def main() -> None:
    server = build_server()

    tuner = MCPTune(
        model="demo-model",
        mcpserver=server,
        seed=42,
    )

    tools = await tuner.discover()
    print(f"Discovered {len(tools)} tools: {[t.name for t in tools]}\n")

    dataset = tuner.build_dataset(tools, samples_per_tool=3)
    print(f"Generated {len(dataset)} rows.\n")

    for row in dataset:
        print(f"{row.tool_name}({row.arguments})")
        print(f"  intent: {row.user_intent}")
        print()


if __name__ == "__main__":
    asyncio.run(main())