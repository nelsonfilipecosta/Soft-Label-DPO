"""Measure the token lengths of the soft-label dataset with the chat template of OLMo 2.

DPO training truncates every sequence (prompt + answer) to `max_length` tokens, which is 1,024 in the smoke
test. TRL 1.14 keeps the start of a sequence, so truncation cuts the end of a long answer and it drops a
pair whose prompt alone reaches `max_length`. This script tokenizes all 10,461 pairs of the soft-label
dataset at its pinned revision the way TRL does, with the chat template applied to the "prompt" alone, to
"prompt + chosen" and to "prompt + rejected" and checks three things:

1. Percentiles of the number of tokens of each sequence and of the longer sequence of each pair.
2. A histogram of the longer sequence of each pair.
3. For each candidate `max_length`, the share of pairs that would be truncated, overall and by bucket,
   and the number of pairs whose prompt alone reaches it.

Usage:
    uv run python experiments/measure_token_lengths.py
"""


import statistics
from collections import Counter

from datasets import concatenate_datasets
from transformers import AutoTokenizer

from soft_label_dpo.data import AGREEMENT_BUCKETS, SPLIT_NAMES, load_soft_label_dataset


MODEL_ID = "allenai/OLMo-2-0425-1B-SFT"
MODEL_REVISION = "0d85a3d037876ce6ac7d4311d994400fc66ac27f"  # same as `revision` in `configs/smoke_test.yaml`

MAX_LENGTHS = (512, 1024, 1536, 2048, 4096)
HISTOGRAM_EDGES = (0, 256, 512, 768, 1024, 1536, 2048, 3072, 4096)


def count_tokens(tokenizer, messages: list[dict], add_generation_prompt: bool = False) -> int:
    """Return the number of tokens of `messages` with the chat template applied."""
    encoding = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
    return len(encoding["input_ids"])


def token_lengths(pair: dict, tokenizer) -> dict:
    """Return the token counts of one pair, so they can be added to the dataset with `Dataset.map`."""
    prompt_chosen = count_tokens(tokenizer, pair["prompt"] + pair["chosen"])
    prompt_rejected = count_tokens(tokenizer, pair["prompt"] + pair["rejected"])
    return {
        "prompt_tokens": count_tokens(tokenizer, pair["prompt"], add_generation_prompt=True),
        "prompt_chosen_tokens": prompt_chosen,
        "prompt_rejected_tokens": prompt_rejected,
        "longer_tokens": max(prompt_chosen, prompt_rejected),
    }


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    splits = load_soft_label_dataset()
    pairs = concatenate_datasets([splits[name] for name in SPLIT_NAMES])
    pairs = pairs.map(token_lengths, fn_kwargs={"tokenizer": tokenizer})
    longer = pairs["longer_tokens"]

    # Check 1: percentiles of each sequence
    print()
    print("Check 1")
    print(f"Tokens per sequence ({len(pairs):,} pairs):")
    print(f"{'sequence':<18} {'median':>7} {'90%':>7} {'95%':>7} {'99%':>7} {'max':>7}")
    columns = {
        "prompt": "prompt_tokens",
        "prompt + chosen": "prompt_chosen_tokens",
        "prompt + rejected": "prompt_rejected_tokens",
        "longer of the two": "longer_tokens",
    }
    for name, column in columns.items():
        values = pairs[column]
        # `quantiles` with n=100 returns the 1st to 99th percentiles
        percentiles = statistics.quantiles(values, n=100, method="inclusive")
        cells = [statistics.median(values), percentiles[89], percentiles[94], percentiles[98], max(values)]
        print(f"{name:<18} " + " ".join(f"{cell:>7,.0f}" for cell in cells))

    # Check 2: histogram of the longer sequence of each pair
    print()
    print("Check 2")
    print("Histogram of the longer sequence of each pair:")
    bins = list(zip(HISTOGRAM_EDGES, HISTOGRAM_EDGES[1:])) + [(HISTOGRAM_EDGES[-1], None)]
    for low, high in bins:
        count = sum(1 for length in longer if length >= low and (high is None or length < high))
        label = f"{low:,}-{high - 1:,}" if high is not None else f"{low:,}+"
        bar = "#" * round(50 * count / len(pairs))
        print(f"{label:>11} {count:>6,} {count / len(pairs):>6.1%} {bar}")

    # Check 3: pairs truncated or dropped at each candidate `max_length`
    print()
    print("Check 3")
    print("Pairs truncated (longer sequence above max_length) and dropped (prompt alone reaches max_length):")
    bucket_sizes = Counter(pairs["bucket"])
    for max_length in MAX_LENGTHS:
        truncated = [bucket for bucket, length in zip(pairs["bucket"], longer) if length > max_length]
        dropped = sum(1 for length in pairs["prompt_tokens"] if length >= max_length)
        by_bucket = Counter(truncated)
        shares = ", ".join(f"{bucket} {by_bucket[bucket] / bucket_sizes[bucket]:.1%}" for bucket in AGREEMENT_BUCKETS)
        print(f"-> max_length {max_length:,}: {len(truncated):,} truncated ({len(truncated) / len(pairs):.1%}; "
              f"{shares}), {dropped:,} dropped")


if __name__ == "__main__":
    main()
