"""Minimal pipeline entrypoint for local experimentation.

Creates a small in-process FastMCP server with a couple of mock tools and
passes it into MCPTune so the rest of the pipeline can discover tools and
generate synthetic data without needing an external MCP process.
"""

import asyncio

from fastmcp import FastMCP

from mcptune import MCPTune


def create_mock_fastmcp_server() -> FastMCP:
	"""Build a small mock FastMCP server for local development."""
	mcp = FastMCP("mock-server")

	@mcp.tool
	def get_weather(city: str) -> str:
		"""Return a mock weather response for a city."""
		return f"Sunny in {city}"

	@mcp.tool
	def add(a: int, b: int) -> int:
		"""Add two numbers."""
		return a + b

	return mcp


async def main() -> None:
	mock_server = create_mock_fastmcp_server()
	tuner = MCPTune(model="something", mcpserver=mock_server)

	tools = await tuner.discover()
	dataset = tuner.build_dataset(tools, 5)

	print(f"Created MCPTune instance: {tuner}")
	for row in dataset:
		print(f"row: {row}")


if __name__ == "__main__":
	asyncio.run(main())