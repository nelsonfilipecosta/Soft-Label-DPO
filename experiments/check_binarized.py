"""Check the majority labels of the soft-label dataset against the "human_overall_binarized" configuration of MultiPref.

Ai2 built the "human_overall_binarized" configuration from the same four votes with their own code, so
both the soft-label and binarized datasets should pick the same chosen answer wherever a pair has a clear
majority. A disagreement is either a difference between the two rules or a bug in the pipeline. The Ai2
pairs where `tie_is_common` is true are left out of the comparison, because Ai2 picked a random winner
for those pairs, and our split pairs (p = 0.5) are also left out of the comparison, because we picked A
by convention.

The answers are matched by text and not by model name, because 27.9% of the pairs compare a model with
itself. This script checks four things on all 10,461 pairs of the soft-label dataset at its pinned revision:

1. Every pair has a binarized pair with the same `comparison_id` and the same prompt text.
2. On the compared pairs, how often our chosen answer is their chosen answer (agree), their rejected
   answer (swapped) or neither of the two.
3. The vote patterns and weighted scores of the agreeing and swapped pairs. The weighted score counts
   "clearly" as 2, "slightly" as 1 and "Tie" as 0 from the side of our chosen answer, to test whether Ai2
   weights "clearly" above "slightly".
4. A few examples of the pairs that match neither answer, printed with `repr` to show hidden whitespace.

Usage:
    uv run python experiments/check_binarized.py
"""


from collections import Counter

from datasets import concatenate_datasets

from soft_label_dpo.data import load_multipref, load_soft_label_dataset


# Weight of each `overall_pref` label, positive for A and negative for B
VOTE_WEIGHTS = {
    "A-is-clearly-better": 2,
    "A-is-slightly-better": 1,
    "Tie": 0,
    "B-is-slightly-better": -1,
    "B-is-clearly-better": -2,
}

OUTCOMES = ("agree", "swapped", "neither", "identical answers")
N_EXAMPLES = 3


def weighted_score(votes: list[str], chosen_is_a: bool) -> int:
    """Return the weighted score of the votes from the side of the chosen answer.

    A positive score means the weighted votes favour our chosen answer, zero means a weighted tie and a
    negative score means they favour our rejected answer.
    """
    score_for_a = sum(VOTE_WEIGHTS[vote] for vote in votes)
    return score_for_a if chosen_is_a else -score_for_a


def outcome(our_chosen: str, their_chosen: str, their_rejected: str) -> str:
    """Return how our chosen answer compares with the binarized pair (one of `OUTCOMES`)."""
    if their_chosen == their_rejected:
        return "identical answers"
    if our_chosen == their_chosen:
        return "agree"
    if our_chosen == their_rejected:
        return "swapped"
    return "neither"


def main():
    splits = load_soft_label_dataset()
    pairs = concatenate_datasets([splits[name] for name in ("train", "validation", "test")])
    binarized = load_multipref("human_overall_binarized")
    # `comparison_id` is unique in the binarized configuration, so each ID maps to one row
    row_of_comparison = {comparison_id: i for i, comparison_id in enumerate(binarized["comparison_id"])}

    # Check 1: every pair has a binarized pair with the same ID and prompt
    matched = [comparison_id in row_of_comparison for comparison_id in pairs["comparison_id"]]
    print()
    print("Check 1")
    print(f"-> {sum(matched):,} of the {len(pairs):,} pairs have a binarized pair with the same `comparison_id`")
    if not all(matched):
        raise SystemExit("Some pairs have no binarized pair, so the IDs don't line up. Stop and investigate.")
    # Reorder the binarized rows to follow our pairs, so the i-th row of both is the same comparison
    binarized = binarized.select([row_of_comparison[comparison_id] for comparison_id in pairs["comparison_id"]])
    same_prompt = sum(
        ours[0]["content"] == theirs
        for ours, theirs in zip(pairs["prompt"], binarized["prompt"])
    )
    print(f"-> {same_prompt:,} of the {len(pairs):,} pairs have the same prompt text")

    # In the binarized configuration, `chosen` and `rejected` are [user message, answer], so the answer is last
    our_chosen = [messages[0]["content"] for messages in pairs["chosen"]]
    their_chosen = [messages[-1]["content"] for messages in binarized["chosen"]]
    their_rejected = [messages[-1]["content"] for messages in binarized["rejected"]]
    outcomes = [outcome(*texts) for texts in zip(our_chosen, their_chosen, their_rejected)]
    compared = [
        bucket != "split" and not tie_is_common
        for bucket, tie_is_common in zip(pairs["bucket"], binarized["tie_is_common"])
    ]
    scores = [
        weighted_score(crowd + expert, is_a)
        for crowd, expert, is_a in zip(pairs["crowd_votes"], pairs["expert_votes"], pairs["chosen_is_a"])
    ]

    # Check 2: agreement on the compared pairs
    counts = Counter(result for result, keep in zip(outcomes, compared) if keep)
    n_compared = sum(compared)
    print()
    print("Check 2")
    print(f"-> {n_compared:,} of the {len(pairs):,} pairs are compared "
          f"(left out: {sum(binarized['tie_is_common']):,} with `tie_is_common`, "
          f"{len(pairs) - n_compared:,} in total with our split pairs)")
    for name in OUTCOMES:
        print(f"-> {name}: {counts[name]:,} ({counts[name] / n_compared:.1%})")

    # Check 3: vote patterns and weighted scores of the agreeing and swapped pairs
    vote_patterns = list(pairs["vote_pattern"])
    print()
    print("Check 3")
    for name in ("agree", "swapped"):
        selected = [i for i, (result, keep) in enumerate(zip(outcomes, compared)) if keep and result == name]
        if not selected:
            print(f"-> {name}: no pairs")
            continue
        patterns = Counter(vote_patterns[i] for i in selected)
        score_counts = Counter(scores[i] for i in selected)
        print(f"-> {name} ({len(selected):,} pairs)")
        print("   vote patterns: " + ", ".join(f"{pattern} {count:,}" for pattern, count in sorted(patterns.items())))
        print("   weighted scores: " + ", ".join(f"{score:+d} {count:,}" for score, count in sorted(score_counts.items())))
    # A swapped pair whose weighted votes still favour our chosen answer fits neither rule and points to a bug
    unexplained = [
        i for i, (result, keep) in enumerate(zip(outcomes, compared))
        if keep and result == "swapped" and scores[i] > 0
    ]
    print(f"-> {len(unexplained):,} swapped pairs have a positive weighted score (the weighted votes favour our chosen answer)")
    for i in unexplained[:N_EXAMPLES]:
        print(f"   {pairs['comparison_id'][i]}: crowd {pairs['crowd_votes'][i]}, expert {pairs['expert_votes'][i]}, "
              f"chosen_is_a {pairs['chosen_is_a'][i]}")

    # Check 4: examples of the pairs that match neither answer
    neither = [i for i, (result, keep) in enumerate(zip(outcomes, compared)) if keep and result == "neither"]
    print()
    print("Check 4")
    print(f"-> {len(neither):,} compared pairs match neither answer (showing up to {N_EXAMPLES})")
    for i in neither[:N_EXAMPLES]:
        print(f"   {pairs['comparison_id'][i]}")
        print(f"   ours:           {our_chosen[i][:200]!r}")
        print(f"   their chosen:   {their_chosen[i][:200]!r}")
        print(f"   their rejected: {their_rejected[i][:200]!r}")


if __name__ == "__main__":
    main()
