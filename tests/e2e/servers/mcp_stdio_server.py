"""A minimal real stdio MCP server, spawned as a subprocess by the e2e test.

Not imported — executed as a separate process over stdin/stdout, which is the
whole point: it exercises the actual subprocess transport, not an in-memory
shortcut.
"""

from fastmcp import FastMCP

server = FastMCP("stdio-e2e")


@server.tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@server.tool
def boom() -> str:
    """Always raises, to exercise tool-level error reporting (is_error=True)."""
    raise ValueError("intentional tool failure")


@server.tool
def echo_env(key: str) -> str:
    """Return an environment variable's value, to prove env injection works."""
    import os

    return os.environ.get(key, "<unset>")


if __name__ == "__main__":
    server.run()  # defaults to stdio transport
