import random
import uuid

from .adapters.fastmcp import FastMCPAdapter
from .sampling.primitive import PrimitiveSampler
from .schema import ToolSpec
from .schema.dataset import DatasetRow


class MCPTune:
    def __init__(self, model: str, mcpserver, adapter=None, seed: int | None = None):
        self.model = model
        self.mcpserver = mcpserver
        self.adapter = adapter or FastMCPAdapter(mcpserver)
        self.seed = seed
        self._rng = random.Random(seed)
        self.sampler = PrimitiveSampler(self._child_rng())

    def _child_rng(self) -> random.Random:
        return random.Random(self._rng.randint(0, 2**32 - 1))

    def _request_id(self) -> str:
        if self.seed is None:
            return str(uuid.uuid4())
        return str(uuid.UUID(int=self._rng.getrandbits(128), version=4))

    async def discover(self) -> list[ToolSpec]:
        """Discover tools from the MCP server as normalized ToolSpec objects."""
        return await self.adapter.discover_tools()

    def build_arguments(self, tool: ToolSpec) -> dict:
        return {param.name: self.sampler.sample(param.schema) for param in tool.parameters}

    def build_mcp_request(self, tool: ToolSpec, arguments: dict) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": self._request_id(),
            "method": "tools/call",
            "params": {
                "name": tool.name,
                "arguments": arguments,
            },
        }

    def build_dataset(self, tools: list[ToolSpec]) -> list[DatasetRow]:
        dataset = []
        for tool in tools:
            arguments = self.build_arguments(tool)
            request = self.build_mcp_request(tool, arguments)
            dataset.append(
                DatasetRow(
                    tool_name=tool.name,
                    arguments=arguments,
                    request=request,
                )
            )
        return dataset

    def train(self, dataset):
        print("[3] Training model...")
        return "trained-model"

    def evaluate(self, model):
        print("[4] Evaluating model...")
        return {"accuracy": 0.9}

    async def run(self):
        tools = await self.discover()
        dataset = self.build_dataset(tools)
        model = self.train(dataset)
        metrics = self.evaluate(model)
        print("Done:", metrics)
        return model, metrics
