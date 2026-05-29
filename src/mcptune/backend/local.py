from mcptune.schema.tools import ToolSpec


class LocalIntentBackend:
    def generate_intent(self, tool: ToolSpec) -> dict:
        # extremely simple heuristic
        return {
            "intent": f"use {tool.name}",
            "arguments_hint": {
                p.name: p.name  # placeholder like "city", "temperature"
                for p in tool.parameters
            },
        }
