"""Real fine-tuning via HuggingFace transformers + PEFT (LoRA).

This is the reference TrainerBackend implementation. It:
  1. Converts DatasetRow -> message format via mcptune.formats
  2. Applies the tokenizer's chat template to produce training text
  3. Wraps the base model in a LoRA adapter via PEFT
  4. Runs a standard HuggingFace Trainer loop
  5. Saves the adapter weights (small) rather than the full model

Defaults are tuned for small models / quick iteration. Override via
the `config` dict passed to train().

A single backend instance trains and saves one model at a time. save()
persists whichever model train() most recently produced; to train
multiple models, instantiate one backend per training job.

Requires mcptune[transformers].
"""

from __future__ import annotations

from pathlib import Path

from mcptune.schema.dataset import DatasetRow

from ..base import TrainerBackend
from ..types import TrainedModel

DEFAULT_CONFIG = {
    "format": "trl",
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

        # 1. Convert DatasetRow -> training-format message rows
        message_rows = convert(dataset, merged["format"])

        # 2. Load tokenizer and base model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.chat_template is None:
            raise ValueError(
                f"Tokenizer for {model_name!r} has no chat_template. "
                "Pick a model with a chat template (Qwen, Llama 3, "
                "SmolLM-Instruct, etc.), or set tokenizer.chat_template "
                "manually before calling train()."
            )

        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
        )

        # 3. Attach LoRA adapter
        lora_config = LoraConfig(
            r=merged["lora_rank"],
            lora_alpha=merged["lora_alpha"],
            lora_dropout=merged["lora_dropout"],
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)

        # 4. Tokenize via chat template
        max_length = merged["max_length"]

        def tokenize(example: dict) -> dict:
            text = tokenizer.apply_chat_template(
                example["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            tokens = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            tokens["labels"] = list(tokens["input_ids"])
            return tokens

        hf_dataset = HFDataset.from_list(message_rows).map(
            tokenize,
            remove_columns=["messages"],
        )

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

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=hf_dataset,
            tokenizer=tokenizer,
        )

        trainer.train()

        # 6. Stash refs for save()
        self._last_trainer = trainer
        self._last_tokenizer = tokenizer

        return TrainedModel(
            backend="transformers",
            metadata={
                "base_model": model_name,
                "num_examples": len(dataset),
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
            raise RuntimeError(
                "No trained model to save. Call train() before save()."
            )
        Path(path).mkdir(parents=True, exist_ok=True)
        self._last_trainer.save_model(path)
        self._last_tokenizer.save_pretrained(path)
        model.model_path = path