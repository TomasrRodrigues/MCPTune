"""Core orchestration for MCPTune pipeline.

This module exposes the `MCPTune` class which coordinates discovery of
tools, synthetic argument generation, intent synthesis, dataset
construction, and delegation to a training backend.
"""

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
    """
    MCPTune is the main orchestration layer of the library.

    It is responsible for:
    - Discovering MCP tools
    - Generating synthetic tool-call arguments
    - Synthesizing natural language user intent
    - Building structured training datasets
    - Delegating training to a backend trainer

    It does NOT:
    - Execute MCP tools
    - Perform model training itself
    - Maintain model weights

    Instead, it acts as a dataset + orchestration pipeline for tool-use ML systems.
    """

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
        """
        Parameters
        ----------
        model:
            Base model identifier used for downstream training.
            This is NOT executed inside MCPTune, it is passed to the trainer backend.

        mcpserver:
            MCP server endpoint or configuration used by the adapter.

        adapter:
            Transport adapter for MCP communication (default: FastMCPAdapter).

        seed:
            Global seed used for deterministic dataset generation.

        semantic_backend:
            Backend used for semantic value generation (e.g., "local", "openai", "none").

        intent_backend:
            Backend used for generating user intents for tool calls.

        intent_model:
            Optional model used by the intent synthesizer.

        intent_llm_call:
            Callable used by LLM-backed intent generation.

        trainer:
            Training backend implementing TrainerBackend interface.
        """
        self.model = model
        self.mcpserver = mcpserver
        self.adapter = adapter or FastMCPAdapter(mcpserver)

        self.seed = 0 if seed is None else seed
        self.rng = random.Random(self.seed)


        # Sampling stack
        self.semantic_backend = semantic_backend
        self.semantic_sampler = SemanticSampler(backend=self.semantic_backend)

        self.sampler = RecursiveSampler(
            self.rng,
            semantic_sampler=self.semantic_sampler,
        )


        # Intent synthesis
        self.intent_synthesizer = IntentSynthesizer(
            backend=intent_backend,
            model=intent_model,
            llm_call=intent_llm_call,
        )


        # Training backend
        self.trainer = trainer



    async def discover(self) -> list[ToolSpec]:
        """
        Discover tools exposed by the MCP server.

        Returns
        -------
        list[ToolSpec]
            Structured representation of available tools.
        """
        return await self.adapter.discover_tools()



    def build_arguments(self, tool: ToolSpec, sample_index: int = 0) -> dict:
        """
        Generate a deterministic argument set for a tool.

        Parameters
        ----------
        tool:
            Tool specification describing name, schema and parameters.

        sample_index:
            Index used to generate multiple distinct samples per tool.

        Returns
        -------
        dict
            Valid arguments matching the tool schema.
        """

        # fallback path (non-recursive sampler)
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
        tools: list[ToolSpec],
        samples_per_tool: int = 1,
    ) -> list[DatasetRow]:
        """
        Generate a full training dataset from MCP tools.

        Each row represents a full tool-use interaction:
        - user intent
        - tool call request
        - tool arguments
        - (optional) execution/response (future)

        Parameters
        ----------
        tools:
            List of MCP tools to sample from.

        samples_per_tool:
            Number of synthetic examples to generate per tool.

        Returns
        -------
        list[DatasetRow]
            Fully constructed dataset ready for formatting/training.
        """
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
        """
        Build a runtime MCP request (non-deterministic ID).

        Used for actual execution, not dataset generation.
        """
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
        """
        Delegate training to configured backend.

        The trainer must implement:
            train(model, dataset, config) -> model
        """
        return self.trainer.train(
            self.model,
            dataset,
            config,
        )


    # TODO: evaluate
    def evaluate(self, model):
        """
        Placeholder evaluation hook (to be expanded in evaluation issues).
        """
        print("[4] Evaluating model...")
        return {"accuracy": 0.9}



    async def run(self):
        """
        Full pipeline:
        discovery -> dataset -> training -> evaluation
        """
        tools = await self.discover()
        dataset = self.build_dataset(tools)
        model = self.train(dataset)
        metrics = self.evaluate(model)

        print("Done:", metrics)
        return model, metrics





    # Deterministic helpers

    def _stable_uuid(self, tool_name: str, arguments: dict) -> str:
        """
        Deterministic request ID based on seed + tool + arguments.
        """
        raw = f"{self.seed}:{tool_name}:{sorted(arguments.items())}".encode()
        digest = hashlib.sha256(raw).digest()
        return str(uuid.UUID(bytes=digest[:16]))



    def _runtime_uuid(self) -> str:
        """
        Non-deterministic UUID for real-time MCP requests.
        """
        return str(uuid.uuid4())



    def _tool_sample_rng(self, tool_name: str, sample_index: int) -> random.Random:
        """
        Deterministic RNG scoped to (seed, tool, sample_index).

        Ensures:
        - reproducibility across runs
        - diversity across samples
        """
        raw = f"{self.seed}:{tool_name}:{sample_index}".encode()
        digest = hashlib.sha256(raw).digest()
        return random.Random(int.from_bytes(digest[:8], "big"))