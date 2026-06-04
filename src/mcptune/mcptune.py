import asyncio
import hashlib
import random
import uuid

from .adapters.fastmcp import FastMCPAdapter
from .sampling.recursive import RecursiveSampler
from .sampling.semantic import SemanticSampler
from .schema import ToolSpec
from .schema.dataset import DatasetRow
from .synthesis.intent import IntentSynthesizer
from .synthesis.answer import AnswerSynthesizer


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
        self.sampler = RecursiveSampler(
            self.rng,
            semantic_sampler=self.semantic_sampler,
        )

        self.intent_synthesizer = IntentSynthesizer(
            backend=intent_backend,
            model=intent_model,
            llm_call=intent_llm_call,
        )

        self.answer_synthesizer = AnswerSynthesizer(
            backend=intent_backend,   # shares the intent backend/model for now (Gap H unifies later)
            model=intent_model,
        )

        self.trainer = trainer
        self._tools: list[ToolSpec] | None = None  # cached by build_dataset for train()

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

    def build_dataset(
        self,
        tools: list[ToolSpec] | None = None,
        samples_per_tool: int = 1,
    ) -> list[DatasetRow]:
        if tools is None:
            tools = asyncio.run(self.discover())

        dataset: list[DatasetRow] = []
        tools = sorted(tools, key=lambda t: t.name)
        self._tools = tools  # remember the tool surface for train()

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

    # replace train()
    def train(self, dataset, config=None, tools=None):
        if self.trainer is None:
            raise RuntimeError(
                "No trainer configured. Pass a TrainerBackend instance via "
                "MCPTune(..., trainer=...)."
            )
        return self.trainer.train(self.model, dataset, config, tools=tools or self._tools)

    def evaluate(self, model):
        # TODO: Phase 5 — evaluation pipeline.
        print("[4] Evaluating model...")
        return {"accuracy": 0.9}

    async def run(self):
        tools = await self.discover()
        dataset = self.build_dataset(tools)
        model = self.train(dataset)
        metrics = self.evaluate(model)
        print("Done:", metrics)
        return model, metrics

    # ---- internal helpers ----

    def _stable_uuid(self, tool_name: str, arguments: dict) -> str:
        raw = f"{self.seed}:{tool_name}:{sorted(arguments.items())}".encode()
        digest = hashlib.sha256(raw).digest()
        return str(uuid.UUID(bytes=digest[:16]))

    def _runtime_uuid(self) -> str:
        return str(uuid.uuid4())

    def _tool_sample_rng(self, tool_name: str, sample_index: int) -> random.Random:
        """Deterministic per-(tool, sample) RNG. Same root seed + tool +
        index always yields the same sequence; different sample_index
        values produce different sequences so multi-sample runs are
        diverse."""
        raw = f"{self.seed}:{tool_name}:{sample_index}".encode()
        digest = hashlib.sha256(raw).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))

    # new method, alongside discover()
    async def execute(self, dataset, *, synthesize_answers: bool = True):
        """Populate response/error by calling each row's tool against the
        server, and (optionally) synthesize the assistant's final answer.

        build_dataset() is offline (sampling + intent). This is the
        online pass: it hits the MCP server and, if enabled, the answer
        LLM. Rows left un-executed still emit valid 2-turn data; running
        this upgrades them to the full call -> result -> answer loop.
        """
        for row in dataset:
            try:
                response = await self.adapter.call_tool(row.tool_name, row.arguments)
                row.response = response
                row.error = None
            except Exception as e:
                row.response = None
                row.error = f"{type(e).__name__}: {e}"

            if synthesize_answers and row.error is None:
                tool = next((t for t in (self._tools or []) if t.name == row.tool_name), None)
                result_text = self._result_to_text(row.response)
                if tool is not None:
                    res = self.answer_synthesizer.synthesize(tool, row.arguments, result_text)
                    row.final_answer = res.answer
                    row.answer_prompt_version = res.prompt_version
        return dataset

    @staticmethod
    def _result_to_text(response) -> str:
        if not isinstance(response, dict):
            return "" if response is None else str(response)
        blocks = response.get("content") or []
        texts = [b.get("text", "") for b in blocks if isinstance(b, dict) and b.get("type") == "text"]
        joined = "\n".join(t for t in texts if t)
        if joined:
            return joined
        sc = response.get("structured_content")
        import json as _json
        return _json.dumps(sc) if sc is not None else ""