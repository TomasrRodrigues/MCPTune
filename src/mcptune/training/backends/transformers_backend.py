"""Real fine-tuning via HuggingFace transformers + PEFT (LoRA).

Reference TrainerBackend. Converts DatasetRow -> the tool_use format
(tools in context + structured calls), renders via the tokenizer chat
template with tools=, attaches a LoRA adapter, runs a Trainer loop, and
saves the adapter (not the full model).

A single instance trains/saves one model at a time. Requires
mcptune[transformers].
"""

from __future__ import annotations

from pathlib import Path

from mcptune.schema.dataset import DatasetRow

from ..base import TrainerBackend
from ..types import TrainedModel

DEFAULT_CONFIG = {
    "format": "tool_use",  # was "trl" — native tool-use is the training default
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "epochs": 1,
    "batch_size": 1,
    "max_length": 512,
    "warmup_steps": 0,
    "logging_steps": 10,
    "save_strategy": "epoch",
}


class TransformersTrainerBackend(TrainerBackend):
    def __init__(self, output_dir: str = "./mcptune-checkpoints"):
        self.output_dir = output_dir
        self._last_trainer = None
        self._last_tokenizer = None

    def train(
        self,
        model_name: str,
        dataset: list[DatasetRow],
        config: dict | None = None,
        tools=None,
    ) -> TrainedModel:
        try:
            import torch
            from datasets import Dataset as HFDataset
            from peft import LoraConfig, TaskType, get_peft_model
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                Trainer,
                TrainingArguments,
            )
        except ImportError as e:
            raise ImportError(
                "TransformersTrainerBackend requires transformers, torch, "
                "peft, accelerate, and datasets. Install with: "
                "pip install mcptune[transformers]"
            ) from e

        from mcptune.formats import convert

        if not dataset:
            raise ValueError("Cannot train on an empty dataset.")

        merged = {**DEFAULT_CONFIG, **(config or {})}

        # 1. Convert DatasetRow -> training-format rows. tool_use needs the
        #    available tools to render into context; convert() raises if the
        #    format requires them and none were passed.
        message_rows = convert(dataset, merged["format"], tools=tools)

        # 2. Tokenizer + base model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.chat_template is None:
            raise ValueError(
                f"Tokenizer for {model_name!r} has no chat_template. Pick a "
                "chat/instruct model (Qwen, Llama 3, SmolLM-Instruct, ...)."
            )

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype)

        # 3. LoRA
        lora_config = LoraConfig(
            r=merged["lora_rank"],
            lora_alpha=merged["lora_alpha"],
            lora_dropout=merged["lora_dropout"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        # 4. Pre-tokenize: render each row's chat template (with tools=) and
        #    tokenize in Python, then build the HF dataset from numeric rows.
        #    Avoids Arrow inferring a schema for the nested messages/tools cols.
        max_length = merged["max_length"]
        tokenized_rows = []
        for ex in message_rows:
            text = tokenizer.apply_chat_template(
                ex["messages"],
                tools=ex.get("tools"),
                tokenize=False,
                add_generation_prompt=False,
            )
            tokens = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
            tokenized_rows.append(
                {
                    "input_ids": tokens["input_ids"],
                    "attention_mask": tokens["attention_mask"],
                    "labels": list(tokens["input_ids"]),
                }
            )
        hf_dataset = HFDataset.from_list(tokenized_rows)

        # 5. Train
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=merged["epochs"],
            per_device_train_batch_size=merged["batch_size"],
            learning_rate=merged["learning_rate"],
            warmup_steps=merged["warmup_steps"],
            logging_steps=merged["logging_steps"],
            save_strategy=merged["save_strategy"],
            report_to="none",
            remove_unused_columns=False,
        )
        # TODO(transformers 5.x, Gap J): tokenizer= -> processing_class=
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=hf_dataset,
            tokenizer=tokenizer,
        )
        trainer.train()

        self._last_trainer = trainer
        self._last_tokenizer = tokenizer

        return TrainedModel(
            backend="transformers",
            metadata={
                "base_model": model_name,
                "num_examples": len(dataset),
                "num_tools": len(tools) if tools else 0,
                "format": merged["format"],
                "lora_rank": merged["lora_rank"],
                "lora_alpha": merged["lora_alpha"],
                "lora_dropout": merged["lora_dropout"],
                "epochs": merged["epochs"],
                "learning_rate": merged["learning_rate"],
            },
        )

    def save(self, model: TrainedModel, path: str) -> None:
        """Persist the most recently trained adapter + tokenizer to `path`."""
        if self._last_trainer is None:
            raise RuntimeError("No trained model to save. Call train() before save().")
        Path(path).mkdir(parents=True, exist_ok=True)
        self._last_trainer.save_model(path)
        self._last_tokenizer.save_pretrained(path)
        model.model_path = path
