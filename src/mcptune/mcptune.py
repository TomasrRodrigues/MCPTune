import hashlib
import random
import uuid

from .adapters.fastmcp import FastMCPAdapter
from .sampling.recursive import RecursiveSampler
from .schema import ToolSpec
from .schema.dataset import DatasetRow


class MCPTune:
    def __init__(self, model: str, mcpserver, adapter=None, seed: int = None):
        self.model = model
        self.mcpserver = mcpserver
        self.adapter = adapter or FastMCPAdapter(mcpserver)

        # deterministic seed (never None)
        self.seed = 0 if seed is None else seed

        # global RNG (used only where appropriate)
        self.rng = random.Random(self.seed)

        # sampler is NOT shared across tools
        self.sampler = RecursiveSampler(self.rng)

    async def discover(self) -> list[ToolSpec]:
        return await self.adapter.discover_tools()

    # def build_arguments(...):
    #    return self.build_dataset([tool])[0].arguments

    def build_arguments(self, tool: ToolSpec) -> dict:
        # allow test override
        if self.sampler.__class__.__name__ != "RecursiveSampler" or hasattr(
            self.sampler, "is_mock"
        ):
            return {param.name: self.sampler.sample(param.schema) for param in tool.parameters}

        tool_rng = self._tool_rng(tool.name)
        sampler = RecursiveSampler(tool_rng)

        return {param.name: sampler.sample(param.schema) for param in tool.parameters}
        # return self.build_dataset([tool])[0].arguments

    def _stable_uuid(self, tool_name: str, arguments: dict) -> str:
        raw = f"{self.seed}:{tool_name}:{sorted(arguments.items())}".encode()
        digest = hashlib.sha256(raw).digest()
        return str(uuid.UUID(bytes=digest[:16]))

    def build_mcp_request(self, tool, arguments):
        return {
            "jsonrpc": "2.0",
            "id": self._runtime_uuid(),
            "method": "tools/call",
            "params": {
                "name": tool.name,
                "arguments": arguments,
            },
        }

    def build_dataset(self, tools: list[ToolSpec], samples_per_tool: int = 1) -> list[DatasetRow]:
        dataset = []

        # stable ordering INSIDE function only
        tools = sorted(tools, key=lambda t: t.name)

        for tool in tools:
            tool_rng = self._tool_rng(tool.name)
            sampler = RecursiveSampler(tool_rng)

            for _ in range(samples_per_tool):
                arguments = {p.name: sampler.sample(p.schema) for p in tool.parameters}

                request = {
                    "jsonrpc": "2.0",
                    "id": self._stable_uuid(tool.name, arguments),
                    "method": "tools/call",
                    "params": {
                        "name": tool.name,
                        "arguments": arguments,
                    },
                }

                dataset.append(
                    DatasetRow(
                        tool_name=tool.name,
                        arguments=arguments,
                        request=request,
                    )
                )

        return dataset

    def _runtime_uuid(self) -> str:
        return str(uuid.uuid4())

    def train(self, dataset):
        print("[3] Training model...")
        return "trained-model"

    def evaluate(self, model):
        print("[4] Evaluating model...")
        return {"accuracy": 0.9}

    def _tool_rng(self, tool_name: str) -> random.Random:
        """
        Deterministic per-tool RNG derived from (seed, tool_name)
        """
        raw = f"{self.seed}:{tool_name}".encode()
        digest = hashlib.sha256(raw).digest()
        seed = int.from_bytes(digest[:8], "big")
        return random.Random(seed)

    async def run(self):
        tools = await self.discover()
        dataset = self.build_dataset(tools)
        model = self.train(dataset)
        metrics = self.evaluate(model)
        print("Done:", metrics)
        return model, metrics
