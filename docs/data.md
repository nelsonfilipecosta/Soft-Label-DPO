# Data preparation

This document describes how the training and evaluation data were built from the human votes of [MultiPref](https://huggingface.co/datasets/allenai/multipref) ([Miranda et al., 2024](https://arxiv.org/abs/2410.19133)). Most choices in preparing the data were tested with a small experiment before they were made. Each section explains what was done, why it was done and which experiment supports it. The [experiments README](../experiments/README.md) has the full list of experiments and results. The [project README](../README.md) describes the research question.

## Overview

The dataset is available on the Hugging Face Hub (it is private for now): [`nelsonfilipecosta/multipref-soft-labels`](https://huggingface.co/datasets/nelsonfilipecosta/multipref-soft-labels).

All code reads the data at one fixed revision, `96819a1cda4e6391735e4078f4fe8e15afe3adec`, which is set as `SOFT_LABEL_DATASET_REVISION` in [`data.py`](../src/soft_label_dpo/data.py). Load it with `load_soft_label_dataset()`:

```python
from soft_label_dpo.data import load_soft_label_dataset

splits = load_soft_label_dataset()  # "train", "validation" and "test"
```

The dataset has 10,461 preference pairs. Each pair has a prompt, two answers and four votes on which answer is better. The pairs are split into train (8,377 pairs), validation (1,042) and test (1,042).

The functions in [`data.py`](../src/soft_label_dpo/data.py) build the dataset in five steps:

1. Load the votes from MultiPref.
2. Turn the four votes of each pair into labels.
3. Make the majority answer of each pair the chosen answer.
4. Split the prompts into train, validation and test.
5. Convert the pairs to the format of TRL and keep only the saved columns.

The script [`build_dataset.py`](../scripts/build_dataset.py) runs all five steps.

## Source

The votes come from the "default" configuration of [MultiPref](https://huggingface.co/datasets/allenai/multipref) ([Miranda et al., 2024](https://arxiv.org/abs/2410.19133)). The dataset is pinned to revision `12910233a0238a997ebe425656e9dfed7b0ff031`, which is set as `MULTIPREF_REVISION` in `data.py`.

- The dataset has **10,461 preference pairs from 5,323 prompts**. Of these, 5,138 (96.5%) prompts appear in two pairs, each with a different pair of answers. The other 185 prompts only appear in one pair ([Agreement buckets per prompt](../experiments/README.md#agreement-buckets-per-prompt)).
- Prompts come **from five sources**: Anthropic HH helpful, Anthropic HH harmless, Chatbot Arena, ShareGPT and WildChat.
- Answers come **from six models**: Tülu 2 7B, Tülu 2 70B, GPT-3.5 Turbo, GPT-4 Turbo, Llama 2 70B Chat and Llama 3 70B Instruct. In 2,919 pairs (27.9%), both answers come from the same model ([Position of the answers](../experiments/README.md#position-of-the-answers)).
- Each preference pair has exactly **four votes** ([MultiPref annotations](../experiments/README.md#multipref-annotations)). Two come from crowdworkers with a Bachelor's degree and two come from experts with at least a graduate degree in the field of the prompt.
- The votes were selected from a **five-point scale**, called `overall_pref`. The vote options were "A-is-clearly-better", "A-is-slightly-better", "Tie", "B-is-slightly-better" and "B-is-clearly-better".

## From votes to labels

The labels are computed from the votes as numbers. Votes for answer A count as 1, ties as 0.5 and votes for answer B as 0. "Slightly" and "clearly" count the same, so the labels measure how many annotators preferred an answer and not how strongly. The four numbers of a pair give five values:

| Value | What it is | Example: B, A, Tie, B |
| --- | --- | --- |
| p | Share of the votes for A | (0 + 1 + 0.5 + 0) / 4 = 0.375 |
| q | Share of the votes for the chosen answer &rArr; max(p, 1 − p) | 0.625 |
| d | Disagreement &rArr; the population variance of the four numbers | 0.171875 |
| vote pattern | The four votes as sorted letters | ABBT |
| bucket | Level of agreement (set by q) | moderate |

The project trains [three conditions](../README.md#approach) on these labels: majority, soft and constant control.

- **q is the training target of the soft condition.** It is the share of annotators who preferred the chosen answer.
- **d separates pairs that q cannot.** For instance, AABB and TTTT both give q = 0.5. However, in AABB the annotators disagree, so d = 0.25, while in TTTT they agree that the two answers are equally good, so d = 0. The [p against d](../experiments/README.md#p-against-d) experiment found that d adds information beyond p for 68.7% of the pairs. The soft labels use only q, so d is kept for the evaluation.
- **p takes only nine values.** Because each pair has four votes, the values of p go from 0 to 1 in steps of 0.125. In consequence, q can only have five values, which are grouped into buckets:

| Bucket | q | Vote patterns | Pairs |
| --- | --- | --- | ---: |
| strong | 0.875 or 1 | AAAA, AAAT, BBBT, BBBB | 3,279 (31.3%) |
| moderate | 0.625 or 0.75 | AAAB, AATT, AABT, ATTT, BTTT, ABBT, BBTT, ABBB | 5,346 (51.1%) |
| split | 0.5 | AABB, ABTT, TTTT | 1,836 (17.6%) |

A split pair has no majority. The buckets are used to balance the split and to report the results by level of agreement.

The confidence of each vote (`overall_confidence`) is not used. Only 4.7% of the votes are `not-confident` or `random-guess` ([MultiPref annotations](../experiments/README.md#multipref-annotations)), so weighting by confidence stays an optional ablation.

Code: `vote_columns` in `data.py`.

## Chosen and rejected answers

Each pair needs a chosen and a rejected answer, because DPO trains the model to prefer one over the other. The chosen answer is the one that most annotators preferred:

- If p > 0.5, A is chosen.
- If p < 0.5, B is chosen. The two answers then swap places, together with their models.
- If p = 0.5, the pair is split and A is chosen by convention.

The column `chosen_is_a` records which answer was chosen, so the original A and B of every pair can be recovered. p always refers to A, which keeps its link to the raw votes. The vote share of the chosen answer is q.

Note that **choosing A by convention for split pairs is harmless**. Split pairs never enter training, because the majority condition has no label for them. Since all three conditions (majority, soft and constant control) train on the same pairs, the split pairs are left out of all three. But they stay in validation and test, because the evaluation needs them. It measures how far the preference of a model is from 50/50, written |σ(βh) − 0.5|. Here σ(βh) is the probability with which the model prefers the chosen answer, computed from its DPO reward margin h. Swapping the two answers turns a preference of 0.55 into 0.45. Both are 0.05 away from 0.5, so the result doesn't change. Choosing A instead of a random answer also keeps every rebuild identical without a seed.

The convention matters only for metrics that need a correct answer. Majority accuracy is one of them, so it is reported for strong and moderate pairs only.

Code: `chosen_and_rejected_columns` in `data.py`.

## Split

The pairs are split into train (80%), validation (10%) and test (10%). Two rules shape the split:

1. **All pairs of a prompt go to the same split.** Otherwise a model could be evaluated on a prompt that it saw in training. So the split assigns prompts and each pair follows its prompt.

2. **Every split gets the same mix of buckets.** The results are reported by bucket, so the splits should have similar bucket shares. However, the two pairs of a prompt can be in different buckets.

The [Agreement buckets per prompt](../experiments/README.md#agreement-buckets-per-prompt) experiment found that 59.2% of the prompts with two pairs have their pairs in different buckets. So each prompt is labelled with the combination of the buckets of its pairs, such as moderate + split. This gives six groups of prompts:

| Buckets of the pairs of the prompt | Prompts | Pairs |
| --- | ---: | ---: |
| moderate + strong | 1,581 (29.7%) | 3,162 (30.2%) |
| moderate | 1,450 (27.2%) | 2,800 (26.8%) |
| moderate + split | 965 (18.1%) | 1,930 (18.4%) |
| strong | 629 (11.8%) | 1,200 (11.5%) |
| split + strong | 498 (9.4%) | 996 (9.5%) |
| split | 200 (3.8%) | 373 (3.6%) |
| total | 5,323 | 10,461 |

The groups with a single bucket also hold the 185 prompts that have only one pair.

Then, each group is shuffled with seed 42 and cut on its own. The first 10% of its prompts (rounded down) go to validation, the next 10% to test and the rest to train. This gives three splits of the following sizes:

| Split | Prompts | Pairs | Strong | Moderate | Split |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 4,263 | 8,377 | 2,631 (31.4%) | 4,274 (51.0%) | 1,472 (17.6%) |
| validation | 530 | 1,042 | 322 (30.9%) | 537 (51.5%) | 183 (17.6%) |
| test | 530 | 1,042 | 326 (31.3%) | 535 (51.3%) | 181 (17.4%) |

The bucket shares differ by about half a point at most between the splits.

Code: `split_by_prompt` and `SPLIT_SEED` in `data.py`.

## Format and columns

The pairs are saved in the conversational format of TRL. The columns `prompt`, `chosen` and `rejected` each hold a list with one message:

```python
{
    "prompt": [{"role": "user", "content": "..."}],
    "chosen": [{"role": "assistant", "content": "..."}],
    "rejected": [{"role": "assistant", "content": "..."}],
}
```

With this format, TRL applies the chat template of the model itself. It also knows where the prompt ends and the answer begins. The texts are copied exactly from MultiPref, without any stripping or cleaning.

The dataset keeps 17 columns:

| Column | Content |
| --- | --- |
| `comparison_id` | ID of the pair in MultiPref, which links each row to the raw data |
| `prompt_id` | ID of the prompt, which keeps the pairs of a prompt in the same split |
| `prompt`, `chosen`, `rejected` | The prompt and the two answers in the conversational format |
| `p`, `q`, `d` | The share of the votes for A, the share for the chosen answer and the disagreement |
| `bucket` | The level of agreement, which can be strong, moderate or split |
| `vote_pattern` | The four votes as sorted letters, such as ABBT |
| `chosen_is_a` | Whether the chosen answer was A in MultiPref |
| `crowd_votes`, `expert_votes` | The two `overall_pref` labels of each group, with "slightly" and "clearly" kept |
| `chosen_model`, `rejected_model` | The models that wrote the two answers |
| `category`, `source` | The task category of the prompt and the dataset it comes from |

The raw columns `text`, `completion_a`, `completion_b`, `model_a` and `model_b` are replaced by the chosen and rejected columns above. Each vote keeps only its `overall_pref`. The rest of each annotation is dropped. This includes the separate preferences on helpfulness, truthfulness and harmlessness, the reasons, the confidence ratings, the annotator IDs and the time spent. The columns `subject_study` and `highest_level_degree` are also dropped. All of the dropped columns stay available in MultiPref through `comparison_id` and the pinned revision.

Code: `to_conversational` and `SAVED_COLUMNS` in `data.py`.

## Building and reproducing

To run the whole pipeline, build the dataset and push it to the Hugging Face Hub, run [`build_dataset.py`](../scripts/build_dataset.py).

```
uv run python scripts/build_dataset.py
```

It needs a Hugging Face token with write access, set as `HF_TOKEN` or with `hf auth login`. The script does five things:

1. It stops if the git working tree has uncommitted changes, so every build matches a commit.
2. It runs the five steps of the pipeline.
3. It pushes the three splits to the Hub as a private dataset.
4. It uploads `build_info.json` next to them.
5. It prints the new revision of the dataset.

`build_info.json` records the MultiPref ID and revision, the split seed, the git commit of the code and the build time. It also records the number of pairs, prompts and pairs per bucket in each split. With it, every training run can be traced back to the exact votes and split that it used. The current revision was built on 2026-11-15 from commit `adb5824`.

**After a rebuild**, copy the printed revision into `SOFT_LABEL_DATASET_REVISION` in `data.py` and commit the change. Every script that calls `load_soft_label_dataset()` then reads the new build. The build is deterministic. The same MultiPref revision, seed and code always give the same splits.

## Checks

### Unit tests

[`test_data.py`](../tests/test_data.py) runs 113 test cases on the functions of `data.py`. They cover the labels, the choice of the chosen answer, the split and the saved columns. Run them with `uv run pytest`.

### Checks of the built dataset

- The [p against d](../experiments/README.md#p-against-d) experiment found that each of the 15 vote patterns gives one value of p and one of d. The pair counts match the raw MultiPref data.
- The [Position of the answers](../experiments/README.md#position-of-the-answers) experiment recovered positions A and B of every pair from `chosen_is_a`. B has the majority in 67.7% of the pairs with a majority, the same share as in the raw MultiPref data. So the answers and their models were swapped correctly.
- The [Token lengths](../experiments/README.md#token-lengths) experiment applied the chat template of OLMo 2 to all 10,461 pairs, as TRL does. So the saved format works with the model.

### Check against "human_overall_binarized"

MultiPref also has a configuration called "human_overall_binarized". In this configuration, Ai2 used the four votes of each pair to pick a chosen answer. The [Binarized labels](../experiments/README.md#binarized-labels) experiment compared these chosen answers with ours. The experiment left out the pairs whose winner is arbitrary in either dataset and it matched the answers by text, because 27.9% of the pairs compare a model with itself. It found three things:

- All 10,461 pairs match by ID, prompt and answer text. No row was misaligned or altered.
- 99.4% of the compared pairs with a B majority have the same chosen answer as in "human_overall_binarized".
- 64.4% of the compared pairs with an A majority **do not have** the same chosen answer as in "human_overall_binarized".

The [Rule of the binarized labels](../experiments/README.md#rule-of-the-binarized-labels) experiment suggests an explanation for this discrepancy. It suggests Ai2 takes the most common of the five labels and gives the win to A only for "A-is-clearly-better". So a pair where most annotators chose "A-is-slightly-better" goes to B. This rule matches the chosen answer of Ai2 on all 5,774 pairs where it applies. It was inferred from the output and not read from the code that Ai2 used. Our labels are correct and the "human_overall_binarized" configuration is not used as ground truth or as a baseline.

## Notes for training and evaluation

- **Training will use 6,905 pairs.** These are the train pairs with a majority. The 1,472 split pairs stay in the saved train split, but the training code will leave them out (see [Chosen and rejected answers](#chosen-and-rejected-answers)). Keeping them in the dataset leaves that choice to the training code.
- **`max_length` will be 2,048 tokens.** At this length, 1.1% of the pairs are truncated. Of these, 31 (0.3% of all pairs) are dropped because their prompt alone fills the limit (see [Token lengths](../experiments/README.md#token-lengths)). The evaluation will use the same length. The first full training run will check that it fits in GPU memory.
- **Chosen answers are longer.** The median prompt with its chosen answer has 391 tokens, against 293 with its rejected answer. DPO may teach the models to write longer answers, so the evaluation will log the length of the generated answers.
- **Annotators favour position B a little.** Over all pairs, B has 67.7% of the majorities, but most of that comes from the stronger models being more often placed in B (see [Position of the answers](../experiments/README.md#position-of-the-answers)). In pairs where both answers come from the same model, B only has 55.4% of the majorities. Training never sees the positions, because every pair becomes a chosen and a rejected answer.
- **Same-model pairs are split more often.** 22.1% of the pairs generated by the same model are split, while only 15.8% of the pairs generated by different models are split (see [Position of the answers](../experiments/README.md#position-of-the-answers)). Annotators disagree more when the two answers are alike. The results on split pairs will also be reported separately for same-model pairs.
- **Pairs with the same q get the same target.** AATT and AAAB both give q = 0.75, although only AAAB has an annotator who preferred the other answer. This is a limitation of the soft labels. d is kept so the evaluation can show it.
- **The test split has 181 split pairs** (85 AABB, 63 ABTT and 33 TTTT). The results on split pairs will be reported overall and by vote pattern, with confidence intervals (see [p against d](../experiments/README.md#p-against-d)).
