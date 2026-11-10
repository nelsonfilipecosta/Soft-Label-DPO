"""Tests for `src/soft_label_dpo/data.py`.

Usage:
    uv run pytest tests/test_data.py -v
"""


import itertools

import pytest

from soft_label_dpo.data import (
    VOTE_LETTERS,
    VOTE_VALUES,
    _check_labels,
    a_is_chosen,
    agreement_bucket,
    chosen_share,
    orient_pair,
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
