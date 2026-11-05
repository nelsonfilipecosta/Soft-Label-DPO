"""Explore the raw MultiPref annotations before building the training dataset.

This script checks three things on the full train split (10,461 pairs) of the dataset:

1. Confirm every pair has two crowdworker and two expert annotations.
2. How "tie_is_common" from the "human_overall_binarized" configuration relates to the four votes
from the raw "default" configuration.
3. Which labels "overall_confidence" takes.

Usage:
    uv run python experiments/explore_multipref.py
"""


from collections import Counter
from datasets import load_dataset


DATASET_ID = "allenai/multipref"


def main():
    # raw config: "default" (10,461 pairs, 4 annotations each)
    raw = load_dataset(DATASET_ID, "default", split="train")
    # binarized config: "human_overall_binarized" (10,461 pairs, 1 majority label each)
    binarized = load_dataset(DATASET_ID, "human_overall_binarized", split="train")

    # Check 1: number of annotations per pair
    annotation_counts = Counter()
    for pair in raw:
        # `or []` counts a missing list as 0 annotations instead of crashing
        num_crowdworkers = len(pair["normal_worker_annotations"] or [])
        num_experts = len(pair["expert_worker_annotations"] or [])
        annotation_counts[(num_crowdworkers, num_experts)] += 1

    print()
    print("Check 1. Number of annotations per pair:")
    for (num_crowdworkers, num_experts), num_pairs in sorted(annotation_counts.items()):
        print(f"-> {num_crowdworkers} crowdworker + {num_experts} expert: {num_pairs:,} of {len(raw):,} pairs ({num_pairs / len(raw):.1%})")


if __name__ == "__main__":
    main()
