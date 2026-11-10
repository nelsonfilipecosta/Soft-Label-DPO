"""Count how the agreement buckets of MultiPref pairs combine within each prompt.

The training data is split by `prompt_id` and stratified by `bucket`, so all pairs of a prompt must go to the
same split. This script checks three things on the full train split (10,461 pairs) of the "default"
configuration:

1. Count the prompts by their number of pairs.
2. Count the prompts with two or more pairs whose pairs are in different buckets.
3. Count every prompt by its combination of buckets, such as "moderate+split".

Usage:
    uv run python experiments/count_prompt_buckets.py
"""


from collections import Counter, defaultdict

from datasets import load_dataset

from soft_label_dpo.data import vote_columns


DATASET_ID = "allenai/multipref"
# Same revision as the MultiPref annotations experiment, so the counts can be compared
DATASET_REVISION = "12910233a0238a997ebe425656e9dfed7b0ff031"


def main():
    raw = load_dataset(DATASET_ID, "default", split="train", revision=DATASET_REVISION)
    pairs = raw.map(vote_columns)

    # Collect the buckets of each prompt's pairs
    buckets_by_prompt = defaultdict(list)
    for prompt_id, bucket in zip(pairs["prompt_id"], pairs["bucket"]):
        buckets_by_prompt[prompt_id].append(bucket)

    # Check 1: number of pairs per prompt
    pair_counts = Counter(len(buckets) for buckets in buckets_by_prompt.values())
    print()
    print("Check 1")
    print(f"Number of pairs per prompt ({len(buckets_by_prompt):,} prompts):")
    for num_pairs, num_prompts in sorted(pair_counts.items()):
        print(f"-> {num_pairs} pair(s): {num_prompts:,} prompts")

    # Check 2: prompts with two or more pairs in different buckets
    multi_pair = [buckets for buckets in buckets_by_prompt.values() if len(buckets) >= 2]
    mixed = [buckets for buckets in multi_pair if len(set(buckets)) >= 2]
    print()
    print("Check 2")
    print(f"-> {len(mixed):,} of the {len(multi_pair):,} prompts with 2+ pairs have pairs in different buckets "
          f"({len(mixed) / len(multi_pair):.1%})")
    print(f"-> these prompts hold {sum(len(buckets) for buckets in mixed):,} of the {len(pairs):,} pairs")

    # Check 3: every prompt by its combination of buckets
    combinations = Counter("+".join(sorted(set(buckets))) for buckets in buckets_by_prompt.values())
    print()
    print("Check 3")
    print("Prompts by combination of buckets:")
    for combination, num_prompts in combinations.most_common():
        print(f"-> {combination}: {num_prompts:,} prompts")


if __name__ == "__main__":
    main()
