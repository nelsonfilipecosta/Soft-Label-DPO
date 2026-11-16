"""Check whether annotators favour position B in MultiPref or the stronger models were placed there.

In the MultiPref annotations experiment, B had the majority in 67.7% of the pairs with a majority. So,
either the annotators favour the answer shown in position B or the stronger models were more often placed
in position B. In a pair where both answers come from the same model, model strength can't favour either
position, so the share of B there tells the two apart. This script checks three things on all 10,461 pairs of
the soft-label dataset at its pinned revision:

1. The share of the majorities that B has over all pairs with a majority.
2. The share of the majorities that B has over the pairs where both answers come from the same model.
3. How often each model appears in position A and in position B.

Usage:
    uv run python experiments/count_position_wins.py
"""


import math
from collections import Counter

from datasets import concatenate_datasets

from soft_label_dpo.data import load_soft_label_dataset


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Return the 95% Wilson confidence interval of the share `successes / n`.

    Unlike the usual "share ± 1.96 standard errors", it stays between 0 and 1 and works for small `n`.
    """
    share = successes / n
    denominator = 1 + z**2 / n
    centre = (share + z**2 / (2 * n)) / denominator
    half_width = z * math.sqrt(share * (1 - share) / n + z**2 / (4 * n**2)) / denominator
    # Clamp because floating-point rounding can put a bound just outside 0 or 1 when the share is 0 or 1
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def print_b_share(label: str, b_wins: int, n: int) -> None:
    """Print the share of the majorities that B has, with its confidence interval."""
    low, high = wilson_interval(b_wins, n)
    print(f"-> {label}: B has the majority in {b_wins:,} of {n:,} pairs ({b_wins / n:.1%}, 95% CI {low:.1%}-{high:.1%})")


def main():
    splits = load_soft_label_dataset()
    pairs = concatenate_datasets([splits[name] for name in ("train", "validation", "test")])

    # Determine which model is in each position
    model_a = [chosen if is_a else rejected
               for chosen, rejected, is_a in zip(pairs["chosen_model"], pairs["rejected_model"], pairs["chosen_is_a"])]
    model_b = [rejected if is_a else chosen
               for chosen, rejected, is_a in zip(pairs["chosen_model"], pairs["rejected_model"], pairs["chosen_is_a"])]

    # B has the majority when the pair has one (not split) and the chosen answer isn't A
    has_majority = [bucket != "split" for bucket in pairs["bucket"]]
    b_wins = [majority and not is_a for majority, is_a in zip(has_majority, pairs["chosen_is_a"])]
    same_model = [a == b for a, b in zip(model_a, model_b)]

    # Check 1: all pairs with a majority
    print()
    print("Check 1")
    print_b_share("all pairs", sum(b_wins), sum(has_majority))

    # Check 2: pairs with a majority whose two answers come from the same model
    print()
    print("Check 2")
    print(f"-> {sum(same_model):,} of the {len(pairs):,} pairs compare a model with itself")
    print_b_share(
        "same model",
        sum(win for win, same in zip(b_wins, same_model) if same),
        sum(majority for majority, same in zip(has_majority, same_model) if same),
    )
    print_b_share(
        "different models",
        sum(win for win, same in zip(b_wins, same_model) if not same),
        sum(majority for majority, same in zip(has_majority, same_model) if not same),
    )

    # Check 3: how often each model appears in each position
    in_a = Counter(model_a)
    in_b = Counter(model_b)
    print()
    print("Check 3")
    print("Appearances of each model in position A and B (all pairs):")
    print(f"{'model':<45} {'A':>6} {'B':>6} {'share B':>8}")
    for model in sorted(in_a.keys() | in_b.keys(), key=lambda model: -(in_a[model] + in_b[model])):
        total = in_a[model] + in_b[model]
        print(f"{model:<45} {in_a[model]:>6,} {in_b[model]:>6,} {in_b[model] / total:>8.1%}")


if __name__ == "__main__":
    main()
