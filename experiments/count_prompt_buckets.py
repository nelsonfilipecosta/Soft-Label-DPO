"""Count how the agreement buckets of MultiPref pairs combine within each prompt.

The training data is split by `prompt_id`, so all pairs of a prompt go to the same split. The split is
also stratified by `bucket`, so each prompt needs one key to stratify by, even when its pairs are in
different buckets. This script checks three things on all 10,461 pairs of the "default" configuration
of MultiPref at its pinned revision:

1. The number of prompts by their number of pairs.
2. The number of prompts with two or more pairs whose pairs are in different buckets.
3. The number of prompts by their combination of buckets, such as "moderate+split".

Usage:
    uv run python experiments/count_prompt_buckets.py
"""


from collections import Counter, defaultdict

from soft_label_dpo.data import load_multipref, vote_columns


def main():
    raw = load_multipref("default")
    pairs = raw.map(vote_columns)

    # Collect the buckets of the pairs of each prompt
    buckets_of_prompt = defaultdict(list)
    for prompt_id, bucket in zip(pairs["prompt_id"], pairs["bucket"]):
        buckets_of_prompt[prompt_id].append(bucket)

    # Check 1: number of pairs per prompt
    pair_counts = Counter(len(buckets) for buckets in buckets_of_prompt.values())
    print()
    print("Check 1")
    print(f"Number of pairs per prompt ({len(buckets_of_prompt):,} prompts):")
    for n_pairs, n_prompts in sorted(pair_counts.items()):
        print(f"-> {n_pairs} pair(s): {n_prompts:,} prompts")

    # Check 2: prompts with two or more pairs in different buckets
    multi_pair = [buckets for buckets in buckets_of_prompt.values() if len(buckets) >= 2]
    mixed = [buckets for buckets in multi_pair if len(set(buckets)) >= 2]
    print()
    print("Check 2")
    print(f"-> {len(mixed):,} of the {len(multi_pair):,} prompts with 2+ pairs have pairs in different buckets "
          f"({len(mixed) / len(multi_pair):.1%})")
    print(f"-> these prompts hold {sum(len(buckets) for buckets in mixed):,} of the {len(pairs):,} pairs")

    # Check 3: every prompt by its combination of buckets
    combinations = Counter("+".join(sorted(set(buckets))) for buckets in buckets_of_prompt.values())
    print()
    print("Check 3")
    print("Prompts by combination of buckets:")
    for combination, n_prompts in combinations.most_common():
        print(f"-> {combination}: {n_prompts:,} prompts")


if __name__ == "__main__":
    main()
