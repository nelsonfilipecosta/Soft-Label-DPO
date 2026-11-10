# Experiments

Each experiment is a one-off check. Each entry gives the question, the setup, the result and the decision it led to.

| Experiment | Question | Decision |
| --- | --- | --- |
| [BOS token in the chat template](#bos-token-in-the-chat-template) | Does the leading `<\|endoftext\|>` in the chat template affect the model's answers? | Keep the template as shipped and use it everywhere. |
| [MultiPref annotations](#multipref-annotations) | Does every pair have four votes and which pairs does `tie_is_common` flag? | Build the labels from the raw votes; use `human_overall_binarized` only for the spot-check. |
| [Agreement buckets per prompt](#agreement-buckets-per-prompt) | Do the pairs of one prompt land in the same agreement bucket? | Stratify the split by each prompt's combination of buckets. |

## BOS token in the chat template

- **Script:** [`compare_bos_token.py`](compare_bos_token.py)
- **Output:** [`compare_bos_token_output.md`](compare_bos_token_output.md)

### Question

The [OLMo 2 1B SFT model card](https://huggingface.co/allenai/OLMo-2-0425-1B-SFT) says the chat template "does NOT have the bos token before the rest". However, the chat template shipped with the tokenizer starts with `{{ bos_token }}`, so every prompt it builds begins with `<|endoftext|>`:

```
<|endoftext|><|user|>
What causes the seasons on Earth?
<|assistant|>
```

Does the leading `{{ bos_token }}` change the model's answers?

### Setup

- Model: `allenai/OLMo-2-0425-1B-SFT` at revision `0d85a3d037876ce6ac7d4311d994400fc66ac27f`.
- Five prompts for different skills: factual, format following, reasoning, opinion and refusal.
- Each prompt run twice: with the template as shipped and with the leading `<|endoftext|>` removed.
- Greedy decoding (`do_sample=False`) with at most 256 new tokens.

### Result

The exact wording of the answers changes with the leading token, but their content is of the same quality on all five prompts.

**Note:** The result was based on five prompts only and the quality judged by eye.

### Decision

Keep the chat template as shipped and use it everywhere: DPO training, held-out evaluation, IFEval and generation.

- **No evidence to justify a change.** No difference in quality was found.
- **The tools use it by default.** TRL and the eval harnesses build prompts with `apply_chat_template`. Overriding the template in each tool risks a silent mismatch.
- **Consistency matters more than the choice.** DPO compares the policy's log-probabilities with the reference model's on the same prompt, so a small effect of the leading token appears on both sides and largely cancels. Mixing formats between training and evaluation would not cancel.

## MultiPref annotations

- **Script:** [`explore_multipref.py`](explore_multipref.py)
- **Output:** printed to the terminal and copied into the result below.

### Question

The soft labels need the four human votes of every pair, which only the `default` configuration of [MultiPref](https://huggingface.co/datasets/allenai/multipref) contains. The `human_overall_binarized` configuration keeps one winner per pair and the dataset card says that when the votes tie it "chooses a random response" and flags the pair in the `tie_is_common` column. Before building the dataset:

1. Does every pair have two crowdworker and two expert annotations?
2. Which vote patterns does `tie_is_common` flag?
3. Which labels does `overall_confidence` take?

### Setup

- Dataset: `allenai/multipref` at revision `12910233a0238a997ebe425656e9dfed7b0ff031`, the `train` split of the `default` and `human_overall_binarized` configurations (10,461 pairs each), matched by `comparison_id`.
- Each annotator's `overall_pref` mapped to A, B or T (tie). "Slightly" and "clearly" both count as a full vote.
- A pair's vote pattern is its four votes in sorted order, such as BBTT, so the order of the annotators doesn't matter.
- p is the share of the four votes for A, with ties counting half.

### Result

**Check 1.** All 10,461 pairs have two crowdworker and two expert annotations, so every pair has four votes.

**Check 2.** Every pair of `default` was found in `human_overall_binarized`.

| Votes | Pairs | Share | `tie_is_common` true | `tie_is_common` false |
| --- | ---: | ---: | ---: | ---: |
| AAAA | 351 | 3.4% | 0 | 351 |
| AAAB | 478 | 4.6% | 0 | 478 |
| AAAT | 425 | 4.1% | 0 | 425 |
| AABB | 755 | 7.2% | 0 | 755 |
| AABT | 684 | 6.5% | 74 | 610 |
| AATT | 411 | 3.9% | 280 | 131 |
| ABBB | 1,141 | 10.9% | 0 | 1,141 |
| ABBT | 987 | 9.4% | 111 | 876 |
| ABTT | 729 | 7.0% | 729 | 0 |
| ATTT | 434 | 4.1% | 434 | 0 |
| BBBB | 1,502 | 14.4% | 0 | 1,502 |
| BBBT | 1,001 | 9.6% | 0 | 1,001 |
| BBTT | 723 | 6.9% | 462 | 261 |
| BTTT | 488 | 4.7% | 488 | 0 |
| TTTT | 352 | 3.4% | 352 | 0 |

- **`tie_is_common` is true for 2,930 pairs (28.0%).** It is always true when Tie is the only most common vote (ABTT, ATTT, BTTT and TTTT) and never true when there is no Tie vote or when three votes agree.
- **It is sometimes true** for AATT (68%), BBTT (64%), AABT (11%) and ABBT (11%), so it depends on more than the number of A, B and T votes. The following rule fits every row of the table: a pair is flagged when Tie is the most common label on the five-point scale, with ties between labels broken by order or at random. This rule is inferred from the counts, not read from Ai2's code.
- **AABB pairs are never flagged** (755, 7.2%), although a 2–2 split has no majority. Their binarized winner cannot follow from the votes and filtering on `tie_is_common` does not remove them.
- **Pairs without a majority** (p = 0.5: AABB, ABTT and TTTT) are 1,836 of the 10,461 pairs (17.6%).
- **B wins far more often than A.** Of the 8,625 pairs with a majority, B has it in 5,842 (67.7%) and A in 2,783 (32.3%). This is not yet explained, but annotators may favour position B or the stronger model may more often be placed in position B.

**Check 3.** Every annotation has one of four labels and none is empty or missing.

| `overall_confidence` | Annotations | Share |
| --- | ---: | ---: |
| absolutely-confident | 28,711 | 68.6% |
| fairly-confident | 11,168 | 26.7% |
| not-confident | 1,614 | 3.9% |
| random-guess | 351 | 0.8% |

### Decision

Build the labels from the four raw votes in `default` and use `human_overall_binarized` only to spot-check them.

- **The raw votes are complete.** Every pair has four votes, so p is computed the same way for all of them.
- **`tie_is_common` doesn't match the soft labels' rule.** It flags pairs with a clear majority under that rule, such as BBTT (p = 0.25), and misses pairs with none, such as AABB.
- **For the spot-check,** leave out pairs where `tie_is_common` is true and pairs with p = 0.5. Match answers by their text and not by model name, because some pairs compare a model with itself.
- **Every vote counts fully, whatever its confidence.** Only 4.7% of votes are `not-confident` or `random-guess`, so weighting by confidence stays an optional ablation. It would need a number for each of the four labels.
- **B's advantage needs a follow-up check.** In pairs where both answers come from the same model, a stronger model cannot be the cause, so a B advantage there would point to annotators favouring a position.

## Agreement buckets per prompt

- **Script:** [`count_prompt_buckets.py`](count_prompt_buckets.py)
- **Output:** printed to the terminal and copied into the result below.

### Question

The data is split 80/10/10 by `prompt_id`, so all pairs of a prompt go to the same split. The split is also stratified by agreement bucket (strong, moderate or split), so every split has similar bucket shares. But MultiPref repeats prompts and the two pairs of one prompt can land in different buckets. A prompt can't be split across training and evaluation, so it needs one key to stratify by:

1. How many pairs does each prompt have?
2. How many prompts have pairs in different buckets?
3. Which combinations of buckets occur?

### Setup

- Dataset: `allenai/multipref` at revision `12910233a0238a997ebe425656e9dfed7b0ff031`, the `train` split of the `default` configuration (10,461 pairs).
- Each pair's bucket comes from `vote_columns` in [`data.py`](../src/soft_label_dpo/data.py), using the rule of the [MultiPref annotations](#multipref-annotations) experiment.
- A prompt's combination is the set of its pairs' buckets, such as moderate + split, so the order of the pairs doesn't matter.

### Result

**Check 1.** The 10,461 pairs come from 5,323 prompts: 5,138 prompts have two pairs and 185 have one. No prompt has more than two.

**Check 2.** 3,044 of the 5,138 prompts with two pairs (59.2%) have their pairs in different buckets. These prompts hold 6,088 of the 10,461 pairs.

**Check 3.**

| Buckets of the prompt's pairs | Prompts |
| --- | ---: |
| moderate + strong | 1,581 |
| moderate | 1,450 |
| moderate + split | 965 |
| strong | 629 |
| split + strong | 498 |
| split | 200 |

- **Agreement depends on the answers, not only on the prompt.** For 498 prompts, one pair is strong and the other is split: the same question got near-unanimous votes for one pair of answers and a 2–2 split for the other.
- **Every combination is large enough to stratify by.** The smallest, split only, still has about 20 prompts for each of the validation and test splits.

### Decision

Split by `prompt_id` and stratify by each prompt's combination of buckets.

- **A single bucket per prompt would hide too much.** Mixed prompts are the majority, so any one bucket per prompt (the first pair's or the lowest-agreement one) would leave the shares of some buckets to chance.
- **The combination balances all three buckets.** Each of the six groups is split 80/10/10 on its own, so every split gets nearly the same share of strong, moderate and split pairs.
- **Check the result after splitting.** The bucket shares of the pairs, not of the prompts, should be similar in all three splits.
