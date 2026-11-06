"""Tests for `src/soft_label_dpo/data.py`.

Usage:
    uv run pytest tests/test_data.py -v
"""


import pytest

from soft_label_dpo.data import VOTE_LETTERS, VOTE_VALUES, _check_labels, vote_share


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
    """Votes A and Tie give p = (1 + 0.5) / 2 = 0.75."""
    assert vote_share(["A-is-clearly-better", "Tie"]) == 0.75


def test_vote_share_counts_slightly_and_clearly_the_same():
    """A "slightly better" vote counts as much as a "clearly better" one."""
    assert vote_share(["A-is-slightly-better", "Tie"]) == vote_share(["A-is-clearly-better", "Tie"])
    assert vote_share(["B-is-slightly-better", "Tie"]) == vote_share(["B-is-clearly-better", "Tie"])


def test_vote_share_checks_the_labels():
    """An unknown label raises the `ValueError` from the help function and not a `KeyError` from the lookup."""
    with pytest.raises(ValueError):
        vote_share(["Tie", "tie"])
