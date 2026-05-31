"""In-process HF generator helpers used for local inference/testing.

This module provides a small convenience wrapper around HuggingFace
transformers to run generation locally when needed. It is not used by
the main MCPTune pipeline by default, but is helpful for experiments
and demonstrations.
"""

import gc

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator:
    """Simple in-process model loader and generator.

    Note: loading large models may consume substantial RAM/VRAM.
    """

    def __init__(self, model_name="Qwen/Qwen2.5-1.5B-Instruct"):
        """Load tokenizer and model from HuggingFace Hub."""
        print(f"Loading tokenizer and model for {model_name}...")

        # Pulls the text-processing tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Pulls the raw model weights and maps them to your hardware automatically (GPU/CPU)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        print("Model loaded successfully into Python process memory!")

    def generate_intent(self, tool, arguments):
        """Produce a user intent string for `tool` and `arguments`."""
        # Constructing the instruction prompt
        prompt = f"""Generate a normal language user prompt to execute this tool and arguments.
        
                Tool: {tool.name} - {tool.description}
                Arguments: {arguments}

                User Prompt:"""

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")

        # Safely move elements to the device ONLY if they are real PyTorch tensors
        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        # Generate tokens
        with torch.no_grad():  # Disable gradient calculation to save memory/speed up
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the numbers back into a human-readable string
        input_ids_structure = inputs["input_ids"]

        # Handle both PyTorch tensors (real run) and nested lists (mocked test)
        if hasattr(input_ids_structure, "shape"):
            prompt_length = input_ids_structure.shape[1]
        else:
            # If it's a mocked nested list like [[1, 2, 3]], get the length of the inner list
            prompt_length = len(input_ids_structure[0])

        # Slice out the prompt tokens from the generated output
        generated_tokens = outputs[0][prompt_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response_text.strip()

    def generate_output(self, input, tool, arguments):
        """Generate a tool output given a user `input`, `tool`, and `arguments`."""
        prompt = f"""
            Based on this input, the tool and its arguments, generate the output
            Tool: {tool.name} - {tool.description}
            Arguments: {arguments}
            User Prompt: {input}
            Output:
        """

        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")

        inputs = {k: v.to(self.model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        input_ids_structure = inputs["input_ids"]

        if hasattr(input_ids_structure, "shape"):
            prompt_length = input_ids_structure.shape[1]
        else:
            prompt_length = len(input_ids_structure[0])

        generated_tokens = outputs[0][prompt_length:]
        response_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response_text.strip()

    def delete(self):
        """Free model and tokenizer resources and run garbage collection."""
        del self.model
        del self.tokenizer

        # Force PyTorch and Python garbage collectors to clear VRAM/RAM immediately
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
