"""Tests for `src/soft_label_dpo/data.py`.

Usage:
    uv run pytest tests/test_data.py -v
"""


import pytest

from soft_label_dpo.data import VOTE_LETTERS, VOTE_VALUES, _check_labels


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
