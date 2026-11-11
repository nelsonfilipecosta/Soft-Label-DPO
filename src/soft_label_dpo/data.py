"""Turn the four human votes on each MultiPref pair into soft labels.

Each pair in the "default" configuration of MultiPref has four votes on which of its two answers, A or B,
is better. Two of the votes come from crowdworkers and two from experts. The `overall_pref` of each vote
is coded as A = 1, Tie (T) = 0.5 or B = 0, with "slightly better" and "clearly better" counting the same.

From these numbers, each pair gets two values:

- p, the share of votes for A, with a tie counting as half a vote for each answer. It is the mean of the
  numbers, so AAAB (three votes for A, one for B) gives p = 0.75. p = 0.5 means there is no majority.

  p(AAAB) = (1+1+1+0)/4 = 0.75
  p(AABB) = (1+1+0+0)/4 = 0.5
  p(TTTT) = (0.5+0.5+0.5+0.5)/4 = 0.5
  p(BBBT) = (0+0+0+0.5)/4 = 0.125

- d, how much the annotators disagree. It is the population variance of the numbers, so it runs from 0,
  when all votes are the same, to 0.25, when half are for A and half are for B. TTTT and AABB both give
  p = 0.5, but TTTT gives d = 0 and AABB gives d = 0.25.

  d(AAAB) = ((1-0.75)^2 + (1-0.75)^2 + (1-0.75)^2 + (0-0.75)^2)/4 = 0.1875
  d(AABB) = ((1-0.5)^2 + (1-0.5)^2 + (0-0.5)^2 + (0-0.5)^2)/4 = 0.25
  d(TTTT) = ((0.5-0.5)^2 + (0.5-0.5)^2 + (0.5-0.5)^2 + (0.5-0.5)^2)/4 = 0
  d(BBBT) = ((0-0.125)^2 + (0-0.125)^2 + (0-0.125)^2 + (0.5-0.125)^2)/4 = 0.046875

Each pair also gets an agreement bucket (strong, moderate or split) from p and a vote pattern, which is
its votes as sorted letters, such as AABT. The functions that take a list of votes accept any number of
them, so they also work on the crowdworker or expert votes alone.
"""


import collections
import random
import statistics

from datasets import Dataset, DatasetDict, load_dataset


# MultiPref pinned to one revision of the dataset so every script reads the same votes
MULTIPREF_ID = "allenai/multipref"
MULTIPREF_REVISION = "12910233a0238a997ebe425656e9dfed7b0ff031"

# Seed of the "train", "validation" and "test" split, so the same split can always be rebuilt
SPLIT_SEED = 42

# Map each `overall_pref` label to a number for p and d ("slightly" and "clearly" count the same)
VOTE_VALUES = {
    "A-is-clearly-better": 1.0,
    "A-is-slightly-better": 1.0,
    "Tie": 0.5,
    "B-is-slightly-better": 0.0,
    "B-is-clearly-better": 0.0,
}

# Map each `overall_pref` label to a letter for the vote pattern, such as "AABT"
VOTE_LETTERS = {
    "A-is-clearly-better": "A",
    "A-is-slightly-better": "A",
    "Tie": "T",
    "B-is-slightly-better": "B",
    "B-is-clearly-better": "B",
}

# The agreement buckets that `agreement_bucket` returns
AGREEMENT_BUCKETS = ("strong", "moderate", "split")


def load_multipref(config: str) -> Dataset:
    """Return the train split of a MultiPref configuration, such as "default", at `MULTIPREF_REVISION`."""
    return load_dataset(MULTIPREF_ID, config, split="train", revision=MULTIPREF_REVISION)


def _check_labels(labels: list[str]) -> None:
    """Raise a `ValueError` if `labels` is empty or holds a label that isn't on the `overall_pref` scale.

    Every function that takes a list of votes calls this first, so a missing or unknown vote stops the
    pipeline instead of silently changing p or d.
    """
    if not labels:
        raise ValueError("Expected at least one vote.")
    for label in labels:
        # `VOTE_VALUES` and `VOTE_LETTERS` have the same labels, so checking `VOTE_VALUES` covers both
        if label not in VOTE_VALUES:
            # `!r` prints the label with quotes, so an empty label (`''`) or a missing one (`None`) stands out
            raise ValueError(f"Unknown vote label {label!r}. Expected one of {list(VOTE_VALUES)}.")


def vote_share(labels: list[str]) -> float:
    """Return p, the share of the votes for A, with a tie counting as half a vote for each answer."""
    _check_labels(labels)
    return statistics.fmean(VOTE_VALUES[label] for label in labels)


def vote_variance(labels: list[str]) -> float:
    """Return d, how much the annotators disagree, as the population variance of the mapped votes."""
    _check_labels(labels)
    # `pvariance` divides by the number of votes, while `variance` would divide by number of votes minus one
    return statistics.pvariance(VOTE_VALUES[label] for label in labels)


def vote_pattern(labels: list[str]) -> str:
    """Return the votes as sorted letters, such as "AABT", so the order of the annotators doesn't matter."""
    _check_labels(labels)
    return "".join(sorted(VOTE_LETTERS[label] for label in labels))


def chosen_share(p: float) -> float:
    """Return q = max(p, 1 - p), the vote share of the majority answer or 0.5 if there is none."""
    if not 0 <= p <= 1:
        raise ValueError(f"Invalid p = {p}. Expected a value between 0 and 1.")
    return max(p, 1 - p)


def agreement_bucket(p: float) -> str:
    """Return the agreement bucket of p: "strong", "moderate" or "split".

    With q = max(p, 1 - p), the bucket is:

    - "strong" if q >= 0.875 (p <= 0.125 or p >= 0.875)
    - "moderate" if 0.625 <= q <= 0.75 (0.25 <= p <= 0.375 or 0.625 <= p <= 0.75)
    - "split" if q = 0.5 (p = 0.5)
    """
    q = chosen_share(p)
    if q >= 0.875:
        return "strong"
    if 0.625 <= q <= 0.75:
        return "moderate"
    if q == 0.5:
        return "split"
    raise ValueError(
        f"No agreement bucket for p = {p}. "
        "Expected q = max(p, 1 - p) >= 0.875, 0.625 <= q <= 0.75 or q = 0.5."
    )


def _annotation_labels(pair: dict, column: str) -> list[str]:
    """Return the `overall_pref` labels in one annotation column of a pair, which must hold exactly two."""
    annotations = pair[column]
    # A missing list (`None`) is an error and not something to count
    count = 0 if annotations is None else len(annotations)
    if count != 2:
        raise ValueError(f"Expected 2 annotations in {column}, but got {count}.")
    return [annotation["overall_pref"] for annotation in annotations]


def vote_columns(pair: dict) -> dict:
    """Return the vote columns of one MultiPref pair, so they can be added to the dataset with `Dataset.map`.

    `p`, `d`, `bucket` and `vote_pattern` come from all four votes. `crowd_votes` and `expert_votes` keep the
    `overall_pref` labels of each group, including "slightly" or "clearly", so the votes of crowdworkers
    and experts can also be studied apart.
    """
    crowd_votes = _annotation_labels(pair, "normal_worker_annotations")
    expert_votes = _annotation_labels(pair, "expert_worker_annotations")
    votes = crowd_votes + expert_votes
    p = vote_share(votes)
    return {
        "p": p,
        "d": vote_variance(votes),
        "bucket": agreement_bucket(p),
        "vote_pattern": vote_pattern(votes),
        "crowd_votes": crowd_votes,
        "expert_votes": expert_votes,
    }


def a_is_chosen(p: float) -> bool:
    """Return `True` if A is the chosen answer (p ≥ 0.5) and `False` otherwise.

    Split pairs (p = 0.5) have no majority, so A is chosen by convention. Swapping the two answers would
    only flip the sign of the model's margin, which leaves the distance of its preference from 50/50 unchanged.
    """
    if not 0 <= p <= 1:
        raise ValueError(f"Invalid p = {p}. Expected a value between 0 and 1.")
    return p >= 0.5


def orient_pair(pair: dict) -> dict:
    """Return the chosen and rejected columns of one pair, so they can be added to the dataset with `Dataset.map`.

    The pair needs `p` from `vote_columns`. The answers and their models are swapped when B is chosen and
    `q` is the vote share of the chosen answer. `chosen_is_a` records the side, so the original A and B can
    always be found again.
    """
    p = pair["p"]
    chosen, rejected = ("a", "b") if a_is_chosen(p) else ("b", "a")
    return {
        "chosen_is_a": chosen == "a",
        "q": chosen_share(p),
        "chosen_completion": pair[f"completion_{chosen}"],
        "rejected_completion": pair[f"completion_{rejected}"],
        "chosen_model": pair[f"model_{chosen}"],
        "rejected_model": pair[f"model_{rejected}"],
    }


def bucket_combination(buckets: list[str]) -> str:
    """Return a prompt's combination of buckets, such as "moderate+strong", to stratify the split by.

    `buckets` holds the bucket of each pair of the prompt. They are sorted and repeats are dropped, so the
    order of the pairs doesn't matter and a prompt whose two pairs are both split gives "split".
    """
    if not buckets:
        raise ValueError("Expected at least one bucket.")
    for bucket in buckets:
        if bucket not in AGREEMENT_BUCKETS:
            raise ValueError(f"Unknown bucket {bucket!r}. Expected one of {list(AGREEMENT_BUCKETS)}.")
    # `set` drops the repeats and `sorted` fixes the order
    return "+".join(sorted(set(buckets)))


def split_prompts(prompt_ids: list[str], seed: int) -> dict[str, str]:
    """Assign each prompt of one group to "train", "validation" or "test" and return them as a dict.

    The prompts are shuffled with `seed`, then "validation" gets the first n // 10, "test" the next n // 10
    and "train" the rest. The same prompts and seed always give the same split, whatever the order of `prompt_ids`.
    """
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("Expected each prompt ID once, but some appear more than once.")
    # Sort first, so the shuffle starts from the same order whatever the order of the rows
    shuffled = sorted(prompt_ids)
    # Use a private generator, so other code that uses `random` can't change this shuffle
    random.Random(seed).shuffle(shuffled)
    # Integer division, so no float rounding can change the size of a split
    n_eval = len(shuffled) // 10
    splits = {}
    for i, prompt_id in enumerate(shuffled):
        if i < n_eval:
            splits[prompt_id] = "validation"
        elif i < 2 * n_eval:
            splits[prompt_id] = "test"
        else:
            splits[prompt_id] = "train"
    return splits


def split_by_prompt(pairs: Dataset, seed: int = SPLIT_SEED) -> DatasetDict:
    """Split the pairs 80/10/10 by prompt, stratified by each prompt's combination of buckets.

    The pairs need `prompt_id` and `bucket`. All pairs of a prompt go to the same split. Each prompt
    is labelled with the combination of its pairs' buckets, such as "moderate+split", which puts the
    prompts in six groups. Each group is shuffled and cut 80/10/10 separately with `split_prompts`.
    So every split gets the same mix of groups and with it nearly the same share of strong, moderate
    and split pairs. Within each split, the pairs keep their original order.
    """
    # Collect the buckets of each prompt's pairs
    buckets_of_prompt = collections.defaultdict(list)
    for prompt_id, bucket in zip(pairs["prompt_id"], pairs["bucket"]):
        buckets_of_prompt[prompt_id].append(bucket)

    # Group the prompts by their combination of buckets
    prompts_of_combination = collections.defaultdict(list)
    for prompt_id, buckets in buckets_of_prompt.items():
        prompts_of_combination[bucket_combination(buckets)].append(prompt_id)

    # Split each combination on its own, then look up the split of every pair through its prompt
    split_of_prompt = {}
    for prompt_ids in prompts_of_combination.values():
        split_of_prompt.update(split_prompts(prompt_ids, seed))
    split_of_pair = [split_of_prompt[prompt_id] for prompt_id in pairs["prompt_id"]]

    return DatasetDict({
        split: pairs.select([i for i, pair_split in enumerate(split_of_pair) if pair_split == split])
        for split in ("train", "validation", "test")
    })
