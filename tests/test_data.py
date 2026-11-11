"""Tests for `src/soft_label_dpo/data.py`.

Usage:
    uv run pytest tests/test_data.py -v
"""


import collections
import itertools

import pytest
from datasets import Dataset

from soft_label_dpo.data import (
    AGREEMENT_BUCKETS,
    VOTE_LETTERS,
    VOTE_VALUES,
    _check_labels,
    a_is_chosen,
    agreement_bucket,
    bucket_combination,
    chosen_share,
    orient_pair,
    split_by_prompt,
    split_prompts,
    vote_columns,
    vote_pattern,
    vote_share,
    vote_variance,
)


# One label per letter, so test votes can be written as patterns such as "AABT"
LABEL_FOR_LETTER = {"A": "A-is-clearly-better", "T": "Tie", "B": "B-is-clearly-better"}

# Votes worked out by hand (pattern, p, d, bucket)
HAND_COMPUTED = [
    ("AAAA", 1.0, 0.0, "strong"),
    ("AAAT", 0.875, 0.046875, "strong"),
    ("AAAB", 0.75, 0.1875, "moderate"),
    ("ATTT", 0.625, 0.046875, "moderate"),
    ("AABB", 0.5, 0.25, "split"),
    ("ABTT", 0.5, 0.125, "split"),
    ("TTTT", 0.5, 0.0, "split"),
    ("BBBB", 0.0, 0.0, "strong"),
]


def labels_from_pattern(pattern: str) -> list[str]:
    """Turn a pattern such as "AABT" into a list of `overall_pref` labels."""
    return [LABEL_FOR_LETTER[letter] for letter in pattern]


def test_mappings_have_the_same_five_labels():
    """Both of the vote mappings must cover the five labels of `overall_pref`."""

    labels = {
        "A-is-clearly-better",
        "A-is-slightly-better",
        "Tie",
        "B-is-slightly-better",
        "B-is-clearly-better",
    }
    assert set(VOTE_VALUES) == labels
    assert set(VOTE_LETTERS) == labels


def test_check_labels_accepts_every_label():
    """Every label of `overall_pref` passes the check without an error."""
    _check_labels(list(VOTE_VALUES))


def test_check_labels_rejects_an_empty_list():
    """A list without votes raises an error."""
    with pytest.raises(ValueError):
        _check_labels([])


@pytest.mark.parametrize(
    "label",
    ["A-is-better", "tie", "", None],
    ids=["typo", "wrong_case", "empty", "missing"],
)
def test_check_labels_rejects_an_unknown_label(label):
    """A label that isn't on the `overall_pref` scale raises an error that names it."""
    with pytest.raises(ValueError) as error:
        _check_labels(["Tie", label])
    assert repr(label) in str(error.value)


@pytest.mark.parametrize("pattern, p", [(pattern, p) for pattern, p, _, _ in HAND_COMPUTED])
def test_vote_share_matches_the_hand_computed_p(pattern, p):
    """p matches the value worked out by hand for each pattern."""
    assert vote_share(labels_from_pattern(pattern)) == p


def test_vote_share_works_with_two_votes():
    """Votes A and Tie give p = (1+0.5)/2 = 0.75."""
    assert vote_share(["A-is-clearly-better", "Tie"]) == 0.75


def test_vote_share_counts_slightly_and_clearly_the_same():
    """A "slightly better" vote counts as much as a "clearly better" one."""
    assert vote_share(["A-is-slightly-better", "Tie"]) == vote_share(["A-is-clearly-better", "Tie"])
    assert vote_share(["B-is-slightly-better", "Tie"]) == vote_share(["B-is-clearly-better", "Tie"])


def test_vote_share_checks_the_labels():
    """An unknown label raises the `ValueError` from the helper function and not a `KeyError` from the lookup."""
    with pytest.raises(ValueError):
        vote_share(["Tie", "tie"])


@pytest.mark.parametrize("pattern, d", [(pattern, d) for pattern, _, d, _ in HAND_COMPUTED])
def test_vote_variance_matches_the_hand_computed_d(pattern, d):
    """d matches the value worked out by hand for each pattern."""
    assert vote_variance(labels_from_pattern(pattern)) == pytest.approx(d)


@pytest.mark.parametrize("pattern, d", [(pattern, d) for pattern, _, d, _ in HAND_COMPUTED])
def test_vote_variance_stays_the_same_when_a_and_b_swap(pattern, d):
    """Swapping A and B, such as AAAT into BBBT, leaves d unchanged."""
    swapped = pattern.translate(str.maketrans("AB", "BA"))
    assert vote_variance(labels_from_pattern(swapped)) == pytest.approx(d)


def test_vote_variance_works_with_two_votes():
    """Votes A and B give d = ((1-0.5)^2 + (0-0.5)^2)/2 = 0.25, the largest possible."""
    assert vote_variance(["A-is-clearly-better", "B-is-clearly-better"]) == pytest.approx(0.25)


def test_vote_variance_checks_the_labels():
    """An unknown label raises the `ValueError` from the helper function and not a `KeyError` from the lookup."""
    with pytest.raises(ValueError):
        vote_variance(["Tie", "tie"])


@pytest.mark.parametrize("pattern", [pattern for pattern, _, _, _ in HAND_COMPUTED])
def test_vote_pattern_matches_the_hand_computed_pattern(pattern):
    """The votes of each pattern give the same pattern back."""
    assert vote_pattern(labels_from_pattern(pattern)) == pattern


def test_vote_pattern_ignores_the_order_of_the_votes():
    """Every order of the votes AABT gives the same pattern."""
    for order in itertools.permutations(labels_from_pattern("AABT")):
        assert vote_pattern(list(order)) == "AABT"


def test_vote_pattern_works_with_two_votes():
    """Votes Tie and A give the pattern AT."""
    assert vote_pattern(["Tie", "A-is-clearly-better"]) == "AT"


def test_vote_pattern_counts_slightly_and_clearly_the_same():
    """A "slightly better" vote gets the same letter as a "clearly better" one."""
    assert vote_pattern(["A-is-slightly-better", "A-is-clearly-better"]) == "AA"
    assert vote_pattern(["B-is-slightly-better", "B-is-clearly-better"]) == "BB"


def test_vote_pattern_checks_the_labels():
    """An unknown label raises the `ValueError` from the helper function and not a `KeyError` from the lookup."""
    with pytest.raises(ValueError):
        vote_pattern(["Tie", "tie"])


@pytest.mark.parametrize("p, q", [(0.0, 1.0), (0.25, 0.75), (0.5, 0.5), (0.75, 0.75)])
def test_chosen_share_is_the_share_of_the_majority_answer(p, q):
    """q is the larger of the two vote shares, whichever answer has the majority."""
    assert chosen_share(p) == q


@pytest.mark.parametrize("p", [-0.125, 1.125])
def test_chosen_share_rejects_a_p_outside_0_and_1(p):
    """A p below 0 or above 1 raises an error."""
    with pytest.raises(ValueError):
        chosen_share(p)


@pytest.mark.parametrize(
    "p, bucket",
    [
        (0.0, "strong"),
        (0.125, "strong"),
        (0.25, "moderate"),
        (0.375, "moderate"),
        (0.5, "split"),
        (0.625, "moderate"),
        (0.75, "moderate"),
        (0.875, "strong"),
        (1.0, "strong"),
    ],
)
def test_agreement_bucket_covers_every_p_from_four_votes(p, bucket):
    """Each of the nine values that p can take with four votes lands in the right bucket."""
    assert agreement_bucket(p) == bucket


@pytest.mark.parametrize("pattern, bucket", [(pattern, bucket) for pattern, _, _, bucket in HAND_COMPUTED])
def test_agreement_bucket_matches_the_hand_computed_bucket(pattern, bucket):
    """The bucket of each pattern's p matches the bucket worked out by hand."""
    assert agreement_bucket(vote_share(labels_from_pattern(pattern))) == bucket


@pytest.mark.parametrize("p", [0.6, 0.8], ids=["between_split_and_moderate", "between_moderate_and_strong"])
def test_agreement_bucket_rejects_a_p_between_buckets(p):
    """A p in a gap between two buckets raises an error."""
    with pytest.raises(ValueError):
        agreement_bucket(p)


def test_agreement_bucket_returns_exactly_the_agreement_buckets():
    """Make sure the `AGREEMENT_BUCKETS` constant and the `agreement_bucket` function don't drift apart."""
    assert {agreement_bucket(k / 8) for k in range(9)} == set(AGREEMENT_BUCKETS)


def test_vote_columns_returns_the_columns_of_a_pair():
    """Crowdworker votes A and Tie and expert votes B and A give the pattern AABT."""
    pair = {
        "normal_worker_annotations": [
            {"overall_pref": "A-is-clearly-better"},
            {"overall_pref": "Tie"},
        ],
        "expert_worker_annotations": [
            {"overall_pref": "B-is-slightly-better"},
            {"overall_pref": "A-is-slightly-better"},
        ],
    }
    assert vote_columns(pair) == {
        "p": 0.625,
        "d": pytest.approx(0.171875),
        "bucket": "moderate",
        "vote_pattern": "AABT",
        "crowd_votes": ["A-is-clearly-better", "Tie"],
        "expert_votes": ["B-is-slightly-better", "A-is-slightly-better"],
    }


@pytest.mark.parametrize(
    "column, annotations",
    [
        ("normal_worker_annotations", [{"overall_pref": "Tie"}]),
        ("expert_worker_annotations", [{"overall_pref": "Tie"}]),
        ("expert_worker_annotations", [{"overall_pref": "Tie"}] * 3),
        ("expert_worker_annotations", None),
    ],
    ids=["one_crowdworker", "one_expert", "three_experts", "missing_experts"],
)
def test_vote_columns_rejects_a_pair_without_two_votes_per_group(column, annotations):
    """A pair raises an error that names the column unless it has two crowdworker and two expert annotations."""
    pair = {
        "normal_worker_annotations": [{"overall_pref": "Tie"}, {"overall_pref": "Tie"}],
        "expert_worker_annotations": [{"overall_pref": "Tie"}, {"overall_pref": "Tie"}],
    }
    pair[column] = annotations
    with pytest.raises(ValueError) as error:
        vote_columns(pair)
    assert column in str(error.value)


@pytest.mark.parametrize(
    "p, expected",
    [(1.0, True), (0.75, True), (0.5, True), (0.25, False), (0.0, False)],
)
def test_a_is_chosen_follows_the_majority(p, expected):
    """A is chosen when it has the majority, B when it has the majority and A by convention when there is none."""
    assert a_is_chosen(p) is expected


@pytest.mark.parametrize("p", [-0.125, 1.5])
def test_a_is_chosen_rejects_a_p_outside_0_and_1(p):
    """A p below 0 or above 1 raises an error."""
    with pytest.raises(ValueError):
        a_is_chosen(p)


@pytest.mark.parametrize(
    "p, expected",
    [
        (
            0.75,
            {
                "chosen_is_a": True,
                "q": 0.75,
                "chosen_completion": "Answer A",
                "rejected_completion": "Answer B",
                "chosen_model": "model-a",
                "rejected_model": "model-b",
            },
        ),
        (
            0.25,
            {
                "chosen_is_a": False,
                "q": 0.75,
                "chosen_completion": "Answer B",
                "rejected_completion": "Answer A",
                "chosen_model": "model-b",
                "rejected_model": "model-a",
            },
        ),
        (
            0.5,
            {
                "chosen_is_a": True,
                "q": 0.5,
                "chosen_completion": "Answer A",
                "rejected_completion": "Answer B",
                "chosen_model": "model-a",
                "rejected_model": "model-b",
            },
        ),
    ],
    ids=["a_majority", "b_majority", "split"],
)
def test_orient_pair_puts_the_majority_answer_in_chosen(p, expected):
    """The majority answer and its model go in the chosen columns and A when there is no majority."""
    pair = {
        "p": p,
        "completion_a": "Answer A",
        "completion_b": "Answer B",
        "model_a": "model-a",
        "model_b": "model-b",
    }
    assert orient_pair(pair) == expected


def test_orient_pair_works_when_both_answers_come_from_the_same_model():
    """The texts still go to the right side when both answers have the same model name."""
    pair = {
        "p": 0.25,
        "completion_a": "Answer A",
        "completion_b": "Answer B",
        "model_a": "gpt-4",
        "model_b": "gpt-4",
    }
    oriented = orient_pair(pair)
    assert oriented["chosen_completion"] == "Answer B"
    assert oriented["rejected_completion"] == "Answer A"


@pytest.mark.parametrize(
    "buckets, combination",
    [
        (["strong"], "strong"),
        (["moderate", "strong"], "moderate+strong"),
        (["strong", "moderate"], "moderate+strong"),
        (["split", "split"], "split"),
        (["strong", "split"], "split+strong"),
    ],
    ids=["one_pair", "two_buckets", "two_buckets_reversed", "same_bucket_twice", "split_and_strong"],
)
def test_bucket_combination_sorts_the_buckets_and_drops_repeats(buckets, combination):
    """The combination doesn't depend on the order of the pairs and names each bucket once."""
    assert bucket_combination(buckets) == combination


@pytest.mark.parametrize("buckets", [[], ["strong", "Strong"]], ids=["no_buckets", "unknown_bucket"])
def test_bucket_combination_rejects_a_missing_or_unknown_bucket(buckets):
    """An empty list or a bucket that `agreement_bucket` can't return raises an error."""
    with pytest.raises(ValueError):
        bucket_combination(buckets)


def make_prompt_ids(n: int) -> list[str]:
    """Return `n` made-up prompt IDs, such as "prompt-007"."""
    return [f"prompt-{i:03d}" for i in range(n)]


def test_split_prompts_assigns_every_prompt_once():
    """Every prompt gets exactly one split and no new prompt appears."""
    prompt_ids = make_prompt_ids(200)
    splits = split_prompts(prompt_ids, seed=0)
    assert sorted(splits) == prompt_ids
    assert set(splits.values()) == {"train", "validation", "test"}


@pytest.mark.parametrize(
    "n, train, validation, test",
    [(200, 160, 20, 20), (25, 21, 2, 2), (5, 5, 0, 0)],
    ids=["multiple_of_10", "rounds_down", "fewer_than_10"],
)
def test_split_prompts_gives_validation_and_test_n_over_10_prompts(n, train, validation, test):
    """Validation and test each get n // 10 prompts and train gets the rest."""
    counts = collections.Counter(split_prompts(make_prompt_ids(n), seed=0).values())
    assert (counts["train"], counts["validation"], counts["test"]) == (train, validation, test)


def test_split_prompts_gives_the_same_split_for_the_same_seed():
    """Two runs with the same seed give the same split."""
    prompt_ids = make_prompt_ids(200)
    assert split_prompts(prompt_ids, seed=0) == split_prompts(prompt_ids, seed=0)


def test_split_prompts_gives_a_different_split_for_a_different_seed():
    """The seed changes which prompts go where."""
    prompt_ids = make_prompt_ids(200)
    assert split_prompts(prompt_ids, seed=0) != split_prompts(prompt_ids, seed=1)


def test_split_prompts_ignores_the_order_of_the_prompts():
    """Reversing the input doesn't change the split, so it doesn't depend on the order of the rows."""
    prompt_ids = make_prompt_ids(200)
    assert split_prompts(prompt_ids[::-1], seed=0) == split_prompts(prompt_ids, seed=0)


def test_split_prompts_rejects_a_repeated_prompt():
    """A prompt ID given twice raises an error instead of being silently assigned once."""
    with pytest.raises(ValueError):
        split_prompts(["prompt-000", "prompt-001", "prompt-000"], seed=0)


def make_pairs() -> Dataset:
    """Return 50 made-up pairs from 30 prompts, where 20 prompts have a strong and a moderate pair and 10
    prompts have one split pair. Each pair has its own `comparison_id`, so it can be followed."""
    rows = []
    for i in range(20):
        rows.append({"comparison_id": f"mixed-{i:02d}-1", "prompt_id": f"mixed-{i:02d}", "bucket": "strong"})
        rows.append({"comparison_id": f"mixed-{i:02d}-2", "prompt_id": f"mixed-{i:02d}", "bucket": "moderate"})
    for i in range(10):
        rows.append({"comparison_id": f"split-{i:02d}-1", "prompt_id": f"split-{i:02d}", "bucket": "split"})
    return Dataset.from_list(rows)


def test_split_by_prompt_returns_the_three_splits_with_all_columns():
    """The result has "train", "validation" and "test" and every split keeps the columns of the pairs."""
    splits = split_by_prompt(make_pairs(), seed=0)
    assert list(splits) == ["train", "validation", "test"]
    for split in splits.values():
        assert split.column_names == ["comparison_id", "prompt_id", "bucket"]


def test_split_by_prompt_puts_every_pair_in_exactly_one_split():
    """No pair is lost or repeated."""
    pairs = make_pairs()
    splits = split_by_prompt(pairs, seed=0)
    split_ids = [comparison_id for split in splits.values() for comparison_id in split["comparison_id"]]
    assert sorted(split_ids) == sorted(pairs["comparison_id"])


def test_split_by_prompt_keeps_each_prompt_in_one_split():
    """No prompt has pairs in two splits."""
    splits = split_by_prompt(make_pairs(), seed=0)
    prompts = [set(split["prompt_id"]) for split in splits.values()]
    for first, second in itertools.combinations(prompts, 2):
        assert first.isdisjoint(second)


def test_split_by_prompt_splits_each_combination_on_its_own():
    """The 20 mixed prompts give 16/2/2 and the 10 split prompts give 8/1/1, so the pairs come out as
    16*2+8=40 in train and 2*2+1=5 in validation and test."""
    splits = split_by_prompt(make_pairs(), seed=0)
    for name, (mixed, split_only) in {"train": (16, 8), "validation": (2, 1), "test": (2, 1)}.items():
        prompts = set(splits[name]["prompt_id"])
        assert sum(prompt.startswith("mixed") for prompt in prompts) == mixed
        assert sum(prompt.startswith("split") for prompt in prompts) == split_only
        assert len(splits[name]) == 2 * mixed + split_only


def test_split_by_prompt_gives_the_same_splits_for_the_same_seed():
    """Two runs with the same seed put the same pairs in each split."""
    first = split_by_prompt(make_pairs(), seed=0)
    second = split_by_prompt(make_pairs(), seed=0)
    for name in ("train", "validation", "test"):
        assert first[name]["comparison_id"] == second[name]["comparison_id"]
