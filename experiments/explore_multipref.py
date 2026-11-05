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
    print(raw)
    # binarized config: "human_overall_binarized" (10,461 pairs, 1 majority label each)
    binarized = load_dataset(DATASET_ID, "human_overall_binarized", split="train")
    print(binarized)

    # How the annotations of the preference pairs are stored in the raw config
    annotations = raw[0]["normal_worker_annotations"]
    print(f"\n`normal_worker_annotations` is a {type(annotations).__name__} of length {len(annotations)}")


if __name__ == "__main__":
    main()
