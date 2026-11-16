"""Compare p against d for the pairs of the soft-label dataset.

p is the share of the four votes of a pair for A (a tie counts half) and d is the population variance of the
votes. Pairs with the same p can differ in d when some votes are ties. For example, AABB and TTTT both
give p = 0.5, but AABB (contested) gives d = 0.25 and TTTT (indifferent) gives d = 0. This script checks
three things on the full dataset (10,461 pairs) at its pinned revision:

1. List every vote pattern with its p, d, bucket and number of pairs.
2. Count the values of d for each p to see for how many pairs d adds information beyond p.
3. Count the vote patterns of the split pairs (p = 0.5) in each split.

Usage:
    uv run python experiments/compare_p_and_d.py
"""


from collections import Counter, defaultdict

from datasets import concatenate_datasets

from soft_label_dpo.data import load_soft_label_dataset


SPLITS = ("train", "validation", "test")


def main():
    splits = load_soft_label_dataset()
    pairs = concatenate_datasets([splits[name] for name in SPLITS])

    # Count the pairs of each (pattern, p, d, bucket)
    rows = Counter(zip(pairs["vote_pattern"], pairs["p"], pairs["d"], pairs["bucket"]))

    # Check 1: every vote pattern from p = 1 down to p = 0 and from low to high d within one p
    print()
    print("Check 1")
    print(f"Vote patterns ({len(pairs):,} pairs):")
    print(f"{'pattern':<8} {'p':>6} {'d':>9}  {'bucket':<9} {'pairs':>6} {'share':>6}")
    for (pattern, p, d, bucket), count in sorted(rows.items(), key=lambda row: (-row[0][1], row[0][2])):
        print(f"{pattern:<8} {p:>6.3f} {d:>9.6f}  {bucket:<9} {count:>6,} {count / len(pairs):>6.1%}")

    # Check 2: the values of d for each p
    d_counts_by_p = defaultdict(Counter)
    for (_, p, d, _), count in rows.items():
        d_counts_by_p[p][d] += count
    print()
    print("Check 2")
    print("Values of d for each p:")
    for p in sorted(d_counts_by_p, reverse=True):
        d_counts = d_counts_by_p[p]
        values = ", ".join(f"{d:.6f}" for d in sorted(d_counts))
        print(f"-> p = {p:.3f}: {len(d_counts)} value(s) of d ({values}) over {sum(d_counts.values()):,} pairs")
    several = sum(sum(d_counts.values()) for d_counts in d_counts_by_p.values() if len(d_counts) > 1)
    print(f"-> {several:,} of the {len(pairs):,} pairs ({several / len(pairs):.1%}) have a p that "
          "comes with more than one value of d")

    # Check 3: the vote patterns of the split pairs in each split
    print()
    print("Check 3")
    print("Vote patterns of the split pairs (p = 0.5) in each split:")
    for name in SPLITS:
        split = splits[name]
        patterns = Counter(
            pattern for pattern, bucket in zip(split["vote_pattern"], split["bucket"]) if bucket == "split"
        )
        counts = ", ".join(f"{pattern} {count:,}" for pattern, count in sorted(patterns.items()))
        print(f"-> {name}: {counts} ({sum(patterns.values()):,} split pairs)")


if __name__ == "__main__":
    main()
