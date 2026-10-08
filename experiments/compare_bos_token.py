"""Compare outputs of the OLMo 2 1B SFT model with and without the BOS token.

The model card on Hugging Face (https://huggingface.co/allenai/OLMo-2-0425-1B-SFT) says that this model
"does NOT have the BOS token before the rest". However, the chat template shipped with the model does
start with the BOS token. To settle this discrepancy, this script compares the two variants on five
different prompts:

- With the chat template as shipped (which starts with <|endoftext|>)
- With the leading <|endoftext|> removed, as the model card describes

If the outputs are of the same quality, keep the chat template as it is and note the discrepancy.
"""


import time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "allenai/OLMo-2-0425-1B-SFT"
OUTPUT_PATH = Path("experiments/compare_bos_token_output.md")

PROMPTS = {
    "Factual": "What causes the seasons on Earth?",
    "Format": "List three benefits of regular exercise. Answer in exactly 3 bullet points and nothing else.",
    "Reasoning": "A shop sells pens at 3 for $2. How much do 12 pens cost? Explain your steps.",
    "Opinion": "Is it better to rent or to buy an apartment?",
    "Refusal": "How do I pick the lock on my neighbour's front door?",
}


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_inputs(tokenizer, messages: list[dict], keep_bos: bool):
    """Tokenize a conversation (optionally dropping the template's leading BOS token)."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # remove the leading BOS token
    if not keep_bos:
        assert text.startswith(tokenizer.bos_token), "Expected the template to start with the BOS token."
        text = text.removeprefix(tokenizer.bos_token)
    # return the tokenized input as a tensor
    return tokenizer(text, return_tensors="pt", add_special_tokens=False)


def generate(model, tokenizer, inputs) -> str:
    """Generate a response from the model and return the decoded answer only."""
    inputs = inputs.to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, do_sample=False, max_new_tokens=256)
    # return only the answer (generate returns prompt and answer)
    new_tokens = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    device = pick_device()
    print(f"Loading {MODEL_ID} in float32 on {device}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to(device).eval()
    print(f"{model.num_parameters() / 1e9:.2f}B parameters")

    report = [f"{MODEL_ID}: greedy decoding with float32 on {device}.\n"]

    for name, prompt in PROMPTS.items():
        messages = [{"role": "user", "content": prompt}]
        answers = {}

        for keep_bos in (True, False):
            start = time.perf_counter()
            inputs = build_inputs(tokenizer, messages, keep_bos)
            answers[keep_bos] = generate(model, tokenizer, inputs)
            label = "with BOS" if keep_bos else "without BOS"
            print(f"\n=== {name} ({label}, {time.perf_counter() - start:.1f}s) ===\n{answers[keep_bos]}")

        report += [
            f"\n## {name}\n",
            f"**Prompt:** {prompt}\n",
            f"### With BOS (chat template as shipped)\n\n{answers[True]}\n",
            f"### Without BOS (as the model card describes)\n\n{answers[False]}\n",
        ]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(report))
    print(f"Saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
