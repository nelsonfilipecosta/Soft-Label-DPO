"""Explore the raw MultiPref annotations before building the training dataset.

This script checks three things on all 10,461 pairs of the "default" configuration of MultiPref at its
pinned revision:

1. Whether every preference pair has two crowdworker and two expert annotations.
2. The vote patterns compared with `tie_is_common` of the "human_overall_binarized" configuration.
3. The labels of `overall_confidence` across all annotations.

Usage:
    uv run python experiments/explore_multipref.py
"""


from collections import Counter

from soft_label_dpo.data import load_multipref


# Map each `overall_pref` label to A, B or T (tie), with "slightly" and "clearly" counting the same
VOTE_LETTERS = {
    "A-is-clearly-better": "A",
    "A-is-slightly-better": "A",
    "Tie": "T",
    "B-is-slightly-better": "B",
    "B-is-clearly-better": "B",
}


def main():
    # Raw config from the "default" configuration (10,461 pairs, 4 annotations each)
    raw = load_multipref("default")
    # Binarized config from the "human_overall_binarized" configuration (10,461 pairs, 1 majority label each)
    binarized = load_multipref("human_overall_binarized")

    # Check 1: number of annotations per pair
    annotation_counts = Counter()
    for pair in raw:
        # `or []` counts a missing list as 0 annotations instead of crashing
        n_crowdworkers = len(pair["normal_worker_annotations"] or [])
        n_experts = len(pair["expert_worker_annotations"] or [])
        annotation_counts[(n_crowdworkers, n_experts)] += 1

    print()
    print("Check 1")
    print("Number of annotations per pair:")
    for (n_crowdworkers, n_experts), n_pairs in sorted(annotation_counts.items()):
        print(f"-> {n_crowdworkers} crowdworker + {n_experts} expert: {n_pairs:,} of {len(raw):,} pairs ({n_pairs / len(raw):.1%})")

    # Check 2: vote patterns against `tie_is_common` from the binarized configuration
    tie_is_common_of_comparison = {pair["comparison_id"]: pair["tie_is_common"] for pair in binarized}
    pattern_counts = Counter()
    n_not_found = 0
    for pair in raw:
        comparison_id = pair["comparison_id"]
        if comparison_id not in tie_is_common_of_comparison:
            n_not_found += 1
            continue
        annotations = (pair["normal_worker_annotations"] or []) + (pair["expert_worker_annotations"] or [])
        # Sort the votes so the order of the annotators doesn't matter (such as "BBTT")
        # and show an unexpected label as "?" instead of crashing
        votes = sorted(VOTE_LETTERS.get(annotation["overall_pref"], "?") for annotation in annotations)
        pattern = "".join(votes)
        pattern_counts[(pattern, tie_is_common_of_comparison[comparison_id])] += 1

    print()
    print("Check 2")
    print("Vote patterns against `tie_is_common`:")
    print(f"   {'votes':<7} {'pairs':>6} {'share':>7} {'true':>6} {'false':>6}")
    patterns = sorted({pattern for pattern, _ in pattern_counts})
    for pattern in patterns:
        n_true = pattern_counts[(pattern, True)]
        n_false = pattern_counts[(pattern, False)]
        n_pairs = n_true + n_false
        print(f"-> {pattern:<7} {n_pairs:>6,} {n_pairs / len(raw):>7.1%} {n_true:>6,} {n_false:>6,}")
    print(f"-> {n_not_found:,} of the {len(raw):,} pairs not found in the binarized configuration")


    # Check 3: count the different labels in `overall_confidence` across all annotations
    confidence_counts = Counter()
    for pair in raw:
        annotations = (pair["normal_worker_annotations"] or []) + (pair["expert_worker_annotations"] or [])
        for annotation in annotations:
            confidence_counts[annotation["overall_confidence"]] += 1

    n_annotations = sum(confidence_counts.values())
    print()
    print("Check 3")
    print("Count of different labels in `overall_confidence`:")
    # `most_common` sorts by count, so the labels don't need to be known in advance
    # `!r` prints them with quotes, so an empty label ('') or a missing one (None) stands out
    for label, n_labels in confidence_counts.most_common():
        print(f"-> {label!r}: {n_labels:,} of {n_annotations:,} annotations ({n_labels / n_annotations:.1%})")


if __name__ == "__main__":
    main()
