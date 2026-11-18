"""Build the MultiPref soft-label dataset and push it to the Hugging Face Hub as a private dataset.

The script runs the whole pipeline of `src/soft_label_dpo/data.py`:

1. Load the "default" configuration of MultiPref at the pinned revision.
2. Add the vote columns (p, d, bucket, ...).
3. Make the majority answer of each pair the chosen answer.
4. Split 80/10/10 by prompt, stratified by the combination of buckets of each prompt.
5. Convert to the conversational format of TRL and keep only `SAVED_COLUMNS`.

It then pushes the "train", "validation" and "test" splits to `SOFT_LABEL_DATASET_ID` and uploads a
`build_info.json` next to them. The file records the MultiPref revision, the split seed and the git
commit of this code, so every training run can be traced back to the exact votes and split it used.

The script refuses to run with uncommitted changes, because the recorded git commit would then not
be the code that built the dataset.

Usage:
    uv run python scripts/build_dataset.py
"""


import collections
import datetime
import json
import subprocess

from datasets import DatasetDict
from huggingface_hub import HfApi

from soft_label_dpo.data import (
    AGREEMENT_BUCKETS,
    MULTIPREF_ID,
    MULTIPREF_REVISION,
    SAVED_COLUMNS,
    SOFT_LABEL_DATASET_ID,
    SPLIT_SEED,
    chosen_and_rejected_columns,
    load_multipref,
    split_by_prompt,
    to_conversational,
    vote_columns,
)


def git_commit() -> str:
    """Return the current git commit or stop if there are uncommitted changes."""
    status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout
    if status.strip():
        raise SystemExit(f"Commit or stash these changes first, so the build can be traced to a commit:\n{status}")
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def build_splits() -> DatasetDict:
    """Return the three splits ready to save."""
    pairs = load_multipref("default").map(vote_columns).map(chosen_and_rejected_columns)
    splits = split_by_prompt(pairs, SPLIT_SEED)
    return splits.map(to_conversational).select_columns(list(SAVED_COLUMNS))


def split_counts(splits: DatasetDict) -> dict:
    """Return the number of pairs, prompts and pairs per bucket in each split."""
    counts = {}
    for name, split in splits.items():
        buckets = collections.Counter(split["bucket"])
        counts[name] = {
            "pairs": len(split),
            "prompts": len(set(split["prompt_id"])),
            "buckets": {bucket: buckets[bucket] for bucket in AGREEMENT_BUCKETS},
        }
    return counts


def main():
    commit = git_commit()

    splits = build_splits()
    build_info = {
        "multipref_id": MULTIPREF_ID,
        "multipref_revision": MULTIPREF_REVISION,
        "split_seed": SPLIT_SEED,
        "git_commit": commit,
        "built_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "splits": split_counts(splits),
    }
    print(json.dumps(build_info, indent=2))

    # `private` only applies when the push creates the repository
    splits.push_to_hub(SOFT_LABEL_DATASET_ID, private=True, commit_message=f"built from git commit `{commit}`")
    upload = HfApi().upload_file(
        path_or_fileobj=json.dumps(build_info, indent=2).encode(),
        path_in_repo="build_info.json",
        repo_id=SOFT_LABEL_DATASET_ID,
        repo_type="dataset",
        commit_message="uploaded `build_info.json`",
    )

    print()
    print(f"Pushed to https://huggingface.co/datasets/{SOFT_LABEL_DATASET_ID}")
    print(f"Revision to pin as `SOFT_LABEL_DATASET_REVISION` in `data.py`: {upload.oid}")


if __name__ == "__main__":
    main()
