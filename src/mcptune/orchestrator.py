from mcptune.backend.base import Backend
from mcptune.intent.generator import IntentGenerator
from mcptune.sampling.recursive import RecursiveSampler


class Orchestrator:
    def __init__(self, backend: Backend, seed: int = 0):
        self.backend = backend
        self.intent = IntentGenerator(seed)
        self.seed = seed

    def run_tool_flow(self, tool, sampler: RecursiveSampler):
        """
        Full pipeline:
        intent → arguments → tool call → response → dataset row
        """

        user_input = self.intent.generate(tool)

        arguments = {p.name: sampler.sample(p.schema) for p in tool.parameters}

        tool_output = self.backend.call_tool(tool.name, arguments)

        return {
            "request": user_input,
            "tool_name": tool.name,
            "arguments": arguments,
            "response": tool_output,
        }
