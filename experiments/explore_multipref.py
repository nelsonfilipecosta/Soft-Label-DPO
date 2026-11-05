"""Explore the raw MultiPref annotations before building the training dataset.

This script checks three things on the full train split (10,461 pairs) from the "default" configuration:

1. Confirm every pair has two crowdworker and two expert annotations.
2. Compare the vote patterns against "tie_is_common" from the "human_overall_binarized" configuration
3. Which labels "overall_confidence" takes.

Usage:
    uv run python experiments/explore_multipref.py
"""


from collections import Counter
from datasets import load_dataset


DATASET_ID = "allenai/multipref"

# Map the five-point scale to A, B or T (tie): "slightly" and "clearly" both count as a full vote
VOTE_LETTERS = {
    "A-is-clearly-better": "A",
    "A-is-slightly-better": "A",
    "Tie": "T",
    "B-is-slightly-better": "B",
    "B-is-clearly-better": "B",
}


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
    print("Check 1")
    print("Number of annotations per pair:")
    for (num_crowdworkers, num_experts), num_pairs in sorted(annotation_counts.items()):
        print(f"-> {num_crowdworkers} crowdworker + {num_experts} expert: {num_pairs:,} of {len(raw):,} pairs ({num_pairs / len(raw):.1%})")


    # Check 2: vote patterns against "tie_is_common" from the binarized configuration
    tie_is_common_by_id = {pair["comparison_id"]: pair["tie_is_common"] for pair in binarized}
    pattern_counts = Counter()
    num_not_found = 0
    for pair in raw:
        comparison_id = pair["comparison_id"]
        if comparison_id not in tie_is_common_by_id:
            num_not_found += 1
            continue
        annotations = (pair["normal_worker_annotations"] or []) + (pair["expert_worker_annotations"] or [])
        # Sort the votes so the order of the annotators doesn't matter (such as "BBTT")
        # and show an unexpected label as "?" instead of crashing
        votes = sorted(VOTE_LETTERS.get(annotation["overall_pref"], "?") for annotation in annotations)
        pattern = "".join(votes)
        pattern_counts[(pattern, tie_is_common_by_id[comparison_id])] += 1

    print()
    print("Check 2")
    print("Vote patterns against 'tie_is_common':")
    print(f"   {'votes':<7} {'pairs':>6} {'share':>7} {'true':>6} {'false':>6}")
    patterns = sorted({pattern for pattern, _ in pattern_counts})
    for pattern in patterns:
        num_true = pattern_counts[(pattern, True)]
        num_false = pattern_counts[(pattern, False)]
        num_pairs = num_true + num_false
        print(f"-> {pattern:<7} {num_pairs:>6,} {num_pairs / len(raw):>7.1%} {num_true:>6,} {num_false:>6,}")
    print(f"-> {num_not_found:,} of the {len(raw):,} pairs not found in the binarized configuration")


if __name__ == "__main__":
    main()
