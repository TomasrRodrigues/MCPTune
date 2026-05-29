# Description grounding

MCPTune's LLM-backed samplers use tool and parameter descriptions to generate more semantically plausible values. This document explains what metadata gets used, where it appears in the prompt, and how to control it.

## What gets used

When `SemanticSampler` builds a prompt for the LLM, it includes:

- The tool name (`ToolSpec.name`)
- The tool description (`ToolSpec.description`)
- For each parameter:
  - The name (`ToolParameter.name`)
  - The schema type and `format` field (if any)
  - The description from the JSONSchema

Parameter descriptions come from each property's `description` field in the tool's input schema. FastMCP populates these from Python docstrings and `Annotated[..., Field(description=...)]` annotations.

## What the grounded prompt looks like

For a tool registered as:

```python
@mcp.tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city."""
    return ...
```

The grounded prompt contains a parameter block of the form:

```
Parameters:
- city (string): (description if available)
- units (string): (description if available)
```

The full template lives at
`src/mcptune/sampling/prompts/grounded_semantic_v1.txt`.

## Missing descriptions

Many MCP servers ship without parameter-level descriptions. When a description is missing the parameter line shows only name and type; generation continues normally, just with less grounding. If you'd rather fail loudly when descriptions are missing — useful for catching schema gaps in your own server — validate the `ToolSpec` before calling the sampler.

## Truncation

Descriptions are truncated to ~800 characters (~200 tokens) per parameter and per tool description, to keep prompt cost bounded. Long text is cut and suffixed with `...`. Adjust the budget by editing `_truncate` in `semantic.py` if your descriptions need more headroom.

## Privacy

Parameter descriptions are public schema content — they ship with the MCP server and are not sensitive. However, prompts may be logged by the LLM backend (Ollama logs requests by default; remote APIs may keep them indefinitely). If your descriptions contain information you'd rather not log, configure the backend's logging accordingly.

## Switching prompt versions

The default since this issue landed is the grounded form:

```python
SemanticSampler()  # uses "grounded_semantic_v1"
```

The previous ungrounded template stays available for ablation experiments and for reproducing datasets generated before grounding existed:

```python
SemanticSampler(prompt_version="semantic_v1")
```

Prompts live in `src/mcptune/sampling/prompts/`. Adding a new one is just dropping a file in that directory and pointing `prompt_version` at its basename.