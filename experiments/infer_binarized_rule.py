"""Infer the rule that Ai2 used to build the "human_overall_binarized" configuration of MultiPref.

The swapped pairs of the binarized labels experiment pointed to a rule that Ai2 takes the most common
of the five `overall_pref` vote labels and maps only "A-is-clearly-better" to A. When several vote labels
are equally common, it picks one at random. The rule was found by looking at the data, so this script tests
it on all pairs. It uses only MultiPref and not our dataset, so a bug in our pipeline can't change the
result. This script checks four things on all 10,461 pairs of MultiPref at its pinned revision:

1. Whether every pair has a binarized pair and the chosen answer of Ai2 matches one of its two answers.
2. The side that Ai2 chose for each most common vote label.
3. Whether the rule predicts every pair with a single most common vote label that isn't "Tie".
4. Whether `tie_is_common` is true only when "Tie" is among the most common vote labels.

Usage:
    uv run python experiments/infer_binarized_rule.py
"""


from collections import Counter

from soft_label_dpo.data import VOTE_VALUES, load_multipref, vote_columns


# The side that Ai2 chooses for each most common vote label under the rule
RULE_SIDES = {
    "A-is-clearly-better": "A",
    "A-is-slightly-better": "B",
    "B-is-slightly-better": "B",
    "B-is-clearly-better": "B",
}

SEVERAL_LABELS = "several vote labels tie"
N_EXAMPLES = 3


def most_common_labels(votes: list[str]) -> list[str]:
    """Return the labels that appear most often in `votes` (sorted), so a tie between labels gives several."""
    counts = Counter(votes)
    top = max(counts.values())
    return sorted(label for label, count in counts.items() if count == top)


def chosen_side(their_chosen: str, completion_a: str, completion_b: str) -> str | None:
    """Return the side ("A" or "B") whose text is the chosen answer of Ai2 or `None` if it can't be told."""
    # When both answers have the same text, the side can't be told apart
    if completion_a == completion_b:
        return None
    if their_chosen == completion_a:
        return "A"
    if their_chosen == completion_b:
        return "B"
    return None


def main():
    pairs = load_multipref("default").map(vote_columns)
    binarized = load_multipref("human_overall_binarized")
    # `comparison_id` is unique in the binarized configuration, so each ID maps to one row
    row_of_comparison = {comparison_id: i for i, comparison_id in enumerate(binarized["comparison_id"])}

    # Check 1: every pair has a binarized pair and the chosen answer of Ai2 matches one of its answers
    matched = [comparison_id in row_of_comparison for comparison_id in pairs["comparison_id"]]
    print()
    print("Check 1")
    print(f"-> {sum(matched):,} of the {len(pairs):,} pairs have a binarized pair with the same `comparison_id`")
    if not all(matched):
        raise SystemExit("Some pairs have no binarized pair, so the IDs don't line up. Stop and investigate.")
    # Reorder the binarized rows to follow the pairs, so the i-th row of both is the same comparison
    binarized = binarized.select([row_of_comparison[comparison_id] for comparison_id in pairs["comparison_id"]])
    # In the binarized configuration, `chosen` is [user message, answer], so the answer is last
    sides = [
        chosen_side(messages[-1]["content"], completion_a, completion_b)
        for messages, completion_a, completion_b in zip(binarized["chosen"], pairs["completion_a"], pairs["completion_b"])
    ]
    side_counts = Counter(sides)
    known = side_counts["A"] + side_counts["B"]
    print(f"-> the side that Ai2 chose is known for {known:,} of the {len(pairs):,} pairs")
    print(f"-> Ai2 chose B in {side_counts['B']:,} of these pairs ({side_counts['B'] / known:.1%})")

    # Check 2: the side that Ai2 chose for each most common vote label
    modes = [most_common_labels(crowd + expert) for crowd, expert in zip(pairs["crowd_votes"], pairs["expert_votes"])]
    sides_of_mode = {}
    for mode, side in zip(modes, sides):
        key = mode[0] if len(mode) == 1 else SEVERAL_LABELS
        sides_of_mode.setdefault(key, Counter())[side] += 1
    print()
    print("Check 2")
    print("Side that Ai2 chose by most common vote label (all pairs):")
    print(f"{'most common vote label':<23} {'A':>6} {'B':>6} {'share B':>8}")
    for key in list(VOTE_VALUES) + [SEVERAL_LABELS]:
        counter = sides_of_mode.get(key, Counter())
        total = counter["A"] + counter["B"]
        share = f"{counter['B'] / total:.1%}" if total else "-"
        print(f"{key:<23} {counter['A']:>6,} {counter['B']:>6,} {share:>8}")

    # Check 3: the rule on the pairs it can predict
    predictable = [
        i for i, (mode, side) in enumerate(zip(modes, sides))
        if len(mode) == 1 and mode[0] != "Tie" and side is not None
    ]
    mismatches = [i for i in predictable if RULE_SIDES[modes[i][0]] != sides[i]]
    print()
    print("Check 3")
    print(f"-> the rule predicts the side that Ai2 chose in {len(predictable) - len(mismatches):,} of the "
          f"{len(predictable):,} pairs with a single most common vote label that isn't \"Tie\"")
    print(f"-> the other {len(pairs) - len(predictable):,} pairs need a random tie-break or have no known side")
    for i in mismatches[:N_EXAMPLES]:
        print(f"   {pairs['comparison_id'][i]}: crowd {pairs['crowd_votes'][i]}, expert {pairs['expert_votes'][i]}, "
              f"Ai2 chose {sides[i]}")

    # Check 4: `tie_is_common` should only be true when "Tie" is among the most common vote labels
    tie_counts = Counter(
        (tie_is_common, "Tie" in mode) for tie_is_common, mode in zip(binarized["tie_is_common"], modes)
    )
    print()
    print("Check 4")
    print(f"-> `tie_is_common` is true in {tie_counts[(True, True)]:,} pairs with \"Tie\" among the most common vote labels")
    print(f"-> `tie_is_common` is true in {tie_counts[(True, False)]:,} pairs without \"Tie\" among the most common vote labels")
    print(f"-> {tie_counts[(False, True)]:,} pairs have \"Tie\" among the most common vote labels but aren't flagged")


if __name__ == "__main__":
    main()
