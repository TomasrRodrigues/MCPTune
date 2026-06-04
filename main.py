from __future__ import annotations

import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "HuggingFaceTB/SmolLM-135M-Instruct"
DEFAULT_ADAPTER_PATH = r".\mcptune-demo-output\finetuned"


def load_model(base_model_name: str, adapter_path: str):
    tokenizer = AutoTokenizer.from_pretrained(adapter_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=dtype,
    )
    base_model.to(device)

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    return tokenizer, model, device


def generate_reply(tokenizer, model, device: str, messages: list[dict[str, str]]):
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}

    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.pad_token_id,
    )

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the fine-tuned adapter.")
    parser.add_argument(
        "--base-model",
        default=DEFAULT_BASE_MODEL,
        help="Base model used during training.",
    )
    parser.add_argument(
        "--adapter-path",
        default=DEFAULT_ADAPTER_PATH,
        help="Path to the saved adapter directory.",
    )
    args = parser.parse_args()

    tokenizer, model, device = load_model(args.base_model, args.adapter_path)

    messages: list[dict[str, str]] = []
    print("Type a message and press Enter. Type 'exit' or 'quit' to stop.\n")

    while True:
        user_text = input("You: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        messages.append({"role": "user", "content": user_text})
        reply = generate_reply(tokenizer, model, device, messages)
        messages.append({"role": "assistant", "content": reply})
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()