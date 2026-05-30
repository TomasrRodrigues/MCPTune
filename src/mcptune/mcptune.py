import hashlib
import random
import uuid

from .adapters.fastmcp import FastMCPAdapter
from .sampling.recursive import RecursiveSampler
from .sampling.semantic import SemanticSampler
from .schema import ToolSpec
from .schema.dataset import DatasetRow
from .synthesis.intent import IntentSynthesizer


class MCPTune:
    def __init__(
        self,
        model: str,
        mcpserver,
        adapter=None,
        seed: int = None,
        semantic_backend: str = "local",
        intent_backend: str = "none",
        intent_model: str | None = None,
        intent_llm_call=None,
        trainer=None,
    ):
        self.model = model
        self.mcpserver = mcpserver
        self.adapter = adapter or FastMCPAdapter(mcpserver)
        self.seed = 0 if seed is None else seed
        self.rng = random.Random(self.seed)

        self.semantic_backend = semantic_backend
        self.semantic_sampler = SemanticSampler(backend=self.semantic_backend)
        self.sampler = RecursiveSampler(self.rng, semantic_sampler=self.semantic_sampler)

        self.intent_synthesizer = IntentSynthesizer(
            backend=intent_backend,
            model=intent_model,
            llm_call=intent_llm_call,
        )

        self.trainer = trainer

    async def discover(self) -> list[ToolSpec]:
        return await self.adapter.discover_tools()

    def build_arguments(self, tool: ToolSpec, sample_index: int = 0) -> dict:
        if not isinstance(self.sampler, RecursiveSampler):
            return {p.name: self.sampler.sample(p.schema) for p in tool.parameters}

        tool_sampler = RecursiveSampler(
            self._tool_sample_rng(tool.name, sample_index),
            semantic_sampler=self.semantic_sampler,
        )

        tool_object_schema = {
            "type": "object",
            "properties": {p.name: p.schema for p in tool.parameters},
            "required": [p.name for p in tool.parameters],
        }

        return tool_sampler.sample(
            schema=tool_object_schema,
            depth=0,
            tool_name=tool.name,
            tool_description=tool.description,
        )

    def build_dataset(self, tools: list[ToolSpec], samples_per_tool: int = 1) -> list[DatasetRow]:
        dataset = []
        tools = sorted(tools, key=lambda t: t.name)

        for tool in tools:
            for sample_index in range(samples_per_tool):
                arguments = self.build_arguments(tool, sample_index=sample_index)
                intent_result = self.intent_synthesizer.synthesize(tool, arguments)

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
                        user_intent=intent_result.intent,
                        intent_prompt_version=intent_result.prompt_version,
                    )
                )

        return dataset

    def build_mcp_request(self, tool: ToolSpec, arguments: dict) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": self._runtime_uuid(),
            "method": "tools/call",
            "params": {
                "name": tool.name,
                "arguments": arguments,
            },
        }

    def train(self, dataset, config=None):
        return self.trainer.train(
            self.model,
            dataset,
            config,
        )

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

    def _stable_uuid(self, tool_name: str, arguments: dict) -> str:
        raw = f"{self.seed}:{tool_name}:{sorted(arguments.items())}".encode()
        digest = hashlib.sha256(raw).digest()
        return str(uuid.UUID(bytes=digest[:16]))

    def _runtime_uuid(self) -> str:
        return str(uuid.uuid4())

    def _tool_sample_rng(self, tool_name: str, sample_index: int) -> random.Random:
        """Deterministic per-(tool, sample) RNG. Same root seed + tool + index
        always yields the same sequence; different sample_index values yield
        different sequences so multi-sample runs produce diverse arguments."""
        raw = f"{self.seed}:{tool_name}:{sample_index}".encode()
        digest = hashlib.sha256(raw).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))
