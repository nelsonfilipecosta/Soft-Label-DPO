"""Tests for `src/soft_label_dpo/data`.

Usage:
    uv run pytest tests/test_data.py -v
"""


from soft_label_dpo.data import VOTE_LETTERS, VOTE_VALUES


def test_mappings_have_the_same_five_labels():
    '''Both of the vote mappings must cover the five labels of `overall_pref`.'''

    labels = {
        "A-is-clearly-better",
        "A-is-slightly-better",
        "Tie",
        "B-is-slightly-better",
        "B-is-clearly-better",
    }
    assert set(VOTE_VALUES) == labels
    assert set(VOTE_LETTERS) == labels
