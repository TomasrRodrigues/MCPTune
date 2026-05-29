import pytest
from fastmcp import FastMCP

from mcptune import MCPTune
from mcptune.adapters.fastmcp import FastMCPAdapter
from mcptune.sampling.primitive import PrimitiveSampler


@pytest.fixture
def fastmcp_server() -> FastMCP:
    """A small in-process FastMCP server with two representative tools."""
    mcp = FastMCP("test-server")

    @mcp.tool
    def get_weather(city: str) -> str:
        """Get the current weather for a city."""
        return f"Weather for {city}"

    @mcp.tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    return mcp


@pytest.fixture
def adapter(fastmcp_server: FastMCP) -> FastMCPAdapter:
    return FastMCPAdapter(fastmcp_server)


@pytest.fixture
def tuner(fastmcp_server: FastMCP) -> MCPTune:
    return MCPTune(model="test-model", mcpserver=fastmcp_server)


@pytest.fixture
def primitive_sampler() -> PrimitiveSampler:
    return PrimitiveSampler()
