# Transport adapters

MCPTune talks to MCP servers through adapters. Every adapter implements the
same `MCPAdapter` interface and returns the **same normalized shapes**, so the
rest of MCPTune (sampling, synthesis, training, evaluation) is identical
regardless of how the server is reached.

All adapters share two guarantees, enforced in `MCPAdapter` itself:

- `discover_tools() -> list[ToolSpec]` - tools normalized to `ToolSpec`,
  preserving the raw JSONSchema (`raw_input_schema`).
- `call_tool(name, arguments) -> dict` - responses normalized to exactly
  `{"content", "structured_content", "is_error"}`.

## Choosing an adapter

| Adapter          | Use when the server is…                              | Construct with |
|------------------|------------------------------------------------------|----------------|
| `FastMCPAdapter` | a FastMCP instance in your own process, or a target FastMCP's `Client` resolves | `FastMCPAdapter(server)` |
| `HTTPAdapter`    | deployed as an HTTP service (production/remote)       | `HTTPAdapter(url, headers=None, timeout=30.0)` |
| `StdioAdapter`   | a local stdio binary (Claude Desktop, most third-party servers) | `StdioAdapter(server_path, env=None)` |

## FastMCPAdapter

In-process or `Client`-resolvable target. The simplest adapter; used in tests
and when you already hold a `FastMCP` instance.

```python
from mcptune.adapters.fastmcp import FastMCPAdapter
adapter = FastMCPAdapter(server)
```

## HTTPAdapter

For MCP servers reachable over HTTP. Uses FastMCP's streamable-http transport.

```python
from mcptune.adapters.http import HTTPAdapter

adapter = HTTPAdapter(
    url="https://mcp.example.com/mcp/",
    headers={"Authorization": "Bearer ..."},  # v1 auth: static headers
    timeout=30.0,                              # per-call, seconds
)
```

- **Auth (v1):** pass a static `headers` dict. OAuth/refresh flows are out of
  scope until a real consumer needs them; FastMCP's `Client`/transport expose
  an `auth=` seam those will attach to later.
- **Timeouts:** 30s per call by default.
- **Retries:** none. Failures bubble to `DatasetRow.error`.

## StdioAdapter

For MCP servers shipped as stdio subprocess binaries - the most common
real-world form. FastMCP launches the server as a child process.

```python
from mcptune.adapters.stdio import StdioAdapter

adapter = StdioAdapter(
    server_path="/path/to/server.py",
    env={"WEATHER_API_KEY": "..."},  # injected into the subprocess
)
```

- **Environment:** stdio servers often need API keys; pass them via `env`.
- **Error model:** a tool returning an error is a normal result with
  `is_error=True`. A subprocess that cannot start or crashes raises - the
  exception propagates rather than masquerading as a tool result, so it lands
  on `DatasetRow.error`.

### Known v1 limitation: a subprocess per call

`StdioAdapter` opens a `Client` (and therefore spawns a fresh subprocess) for
**every** `discover_tools` and `call_tool`. This is intentionally simple for
v1, at the cost of repeated process startup. Persistent session reuse is a
planned optimization and is deferred until the adapter contract suite is in
place. For large dataset generation against a stdio server, expect process-spawn
overhead to dominate; HTTP or in-process targets avoid it.

## Response shape (all adapters)

```python
{
    "content": [...],            # raw response blocks
    "structured_content": {...}, # parsed payload, if the tool returns one
    "is_error": False,           # True for tool-level errors
}
```

This shape is produced by `MCPAdapter._normalize_response` and is identical
across every adapter - that's the contract the transport abstraction exists to
guarantee.