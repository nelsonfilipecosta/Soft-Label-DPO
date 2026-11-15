# Experiments

Each experiment is a one-off check. Each entry gives the question, the setup, the result and the decision it led to.

| Experiment | Question | Decision |
| --- | --- | --- |
| [BOS token in the chat template](#bos-token-in-the-chat-template) | Does the leading `<\|endoftext\|>` in the chat template affect the model's answers? | Keep the template as shipped and use it everywhere. |
| [MultiPref annotations](#multipref-annotations) | Does every pair have four votes and which pairs does `tie_is_common` flag? | Build the labels from the raw votes and use `human_overall_binarized` only for the spot-check. |
| [Agreement buckets per prompt](#agreement-buckets-per-prompt) | Do the pairs of one prompt land in the same agreement bucket? | Stratify the split by each prompt's combination of buckets. |
| [p against d](#p-against-d) | For how many pairs does d add information beyond p, and how are split pairs divided by vote pattern? | Report split pairs overall and by vote pattern, with confidence intervals. |
| [Token lengths](#token-lengths) | How many pairs does each `max_length` truncate or drop? | Use `max_length` 2,048 in training and evaluation, if it fits in memory. |
| [Position of the answers](#position-of-the-answers) | Why does B win 67.7% of the majorities: do annotators favour position B or were stronger models placed there? | Keep the data as it is and note the small position bias and the non-random placement in the write-up. |

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
- **B wins far more often than A.** Of the 8,625 pairs with a majority, B has it in 5,842 (67.7%) and A in 2,783 (32.3%). Annotators may favour position B or the stronger model may more often be placed in position B. The [Position of the answers](#position-of-the-answers) experiment finds that it is mostly the placement of the models.

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
- **B's advantage needs a follow-up check.** In pairs where both answers come from the same model, a stronger model cannot be the cause, so a B advantage there would point to annotators favouring a position. See [Position of the answers](#position-of-the-answers).

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

## p against d

- **Script:** [`tabulate_p_and_d.py`](tabulate_p_and_d.py)
- **Output:** printed to the terminal and copied into the result below.

### Question

Each pair gets two numbers from its four votes: p, the share for A with ties counting half, and d, the population variance of the votes. Pairs with the same p can differ in d when some votes are ties. AABB and TTTT both give p = 0.5, but AABB is contested (d = 0.25) and TTTT is indifferent (d = 0). The soft labels use only q = max(p, 1 − p), while the evaluation can also look at d. Before the evaluation:

1. Does each vote pattern give one value of p and one of d?
2. For how many pairs does d add information beyond p?
3. How many split pairs of each vote pattern does each split have?

### Setup

- Dataset: `nelsonfilipecosta/multipref-soft-labels` at revision `96819a1cda4e6391735e4078f4fe8e15afe3adec`, all three splits (10,461 pairs). It was built by [`build_dataset.py`](../scripts/build_dataset.py) from `allenai/multipref` at revision `12910233a0238a997ebe425656e9dfed7b0ff031`.
- p, d and the bucket come from `vote_columns` in [`data.py`](../src/soft_label_dpo/data.py): A = 1, Tie = 0.5 and B = 0, with "slightly" and "clearly" counting the same.

### Result

**Check 1.** Each of the 15 vote patterns gives one value of p and one of d, and the pair counts match the [MultiPref annotations](#multipref-annotations) experiment.

| Votes | p | d | Bucket | Pairs | Share |
| --- | ---: | ---: | --- | ---: | ---: |
| AAAA | 1.000 | 0.000000 | strong | 351 | 3.4% |
| AAAT | 0.875 | 0.046875 | strong | 425 | 4.1% |
| AATT | 0.750 | 0.062500 | moderate | 411 | 3.9% |
| AAAB | 0.750 | 0.187500 | moderate | 478 | 4.6% |
| ATTT | 0.625 | 0.046875 | moderate | 434 | 4.1% |
| AABT | 0.625 | 0.171875 | moderate | 684 | 6.5% |
| TTTT | 0.500 | 0.000000 | split | 352 | 3.4% |
| ABTT | 0.500 | 0.125000 | split | 729 | 7.0% |
| AABB | 0.500 | 0.250000 | split | 755 | 7.2% |
| BTTT | 0.375 | 0.046875 | moderate | 488 | 4.7% |
| ABBT | 0.375 | 0.171875 | moderate | 987 | 9.4% |
| BBTT | 0.250 | 0.062500 | moderate | 723 | 6.9% |
| ABBB | 0.250 | 0.187500 | moderate | 1,141 | 10.9% |
| BBBT | 0.125 | 0.046875 | strong | 1,001 | 9.6% |
| BBBB | 0.000 | 0.000000 | strong | 1,502 | 14.4% |

**Check 2.** d adds information beyond p for 7,182 of the 10,461 pairs (68.7%): every moderate and every split pair. Each p of a strong pair (0, 0.125, 0.875 and 1) comes with one value of d, each p of a moderate pair with two and p = 0.5 with three.

- **Within a moderate p, d tells ties from opposing votes.** AATT and AAAB both give p = 0.75. In AATT, the two votes that don't prefer A are ties, so d = 0.0625. In AAAB, one annotator prefers B, so d = 0.1875. The soft labels give both the same target, q = 0.75.
- **d doesn't follow the level of agreement.** ATTT has less agreement than AAAB (q = 0.625 against 0.75) but also a lower d (0.046875 against 0.1875), because none of its votes prefers the other answer.

**Check 3.** The split pairs are stratified as one bucket, so their vote patterns are left to chance in each split.

| Split | AABB | ABTT | TTTT | Split pairs |
| --- | ---: | ---: | ---: | ---: |
| train | 591 (40.1%) | 596 (40.5%) | 285 (19.4%) | 1,472 |
| validation | 79 (43.2%) | 70 (38.3%) | 34 (18.6%) | 183 |
| test | 85 (47.0%) | 63 (34.8%) | 33 (18.2%) | 181 |

- **The test split has more contested pairs than train** (47.0% AABB against 40.1%). Every condition is evaluated on the same test pairs, so this changes the overall value on split pairs but not the comparison between conditions.
- **The test split has only 33 TTTT pairs,** so a result for them alone can only show a large difference.

### Decision

Report the results on split pairs overall and by vote pattern, each with a confidence interval.

- **The overall value on the 181 split pairs of the test split is the main result.** It has the most pairs and is the same for every condition.
- **The breakdown by vote pattern is secondary.** Contested (AABB) and indifferent (TTTT) pairs may be treated differently by the models, but with 85 and 33 pairs only large differences will show, so the confidence intervals must be reported with it.
- **Keep d in the dataset.** The soft labels use only q, so pairs with the same q but different d, such as AATT and AAAB, get the same target. This is a limitation to note in the write-up, and d is what makes it visible.

## Token lengths

- **Script:** [`measure_token_lengths.py`](measure_token_lengths.py)
- **Output:** printed to the terminal and copied into the result below.

### Question

DPO training cuts every sequence (prompt + answer) to `max_length` tokens, which is 1,024 in the smoke test. TRL 1.14 keeps the start of a sequence, so a long answer loses its end, and it drops a pair from training when its prompt alone reaches `max_length`. Before choosing `max_length` for the real runs:

1. How long are the prompts and the two sequences of each pair?
2. How are the lengths of the longer sequence of each pair distributed?
3. How many pairs does each candidate `max_length` truncate or drop, and does it hit one bucket more than the others?

### Setup

- Dataset: `nelsonfilipecosta/multipref-soft-labels` at revision `96819a1cda4e6391735e4078f4fe8e15afe3adec`, all three splits (10,461 pairs).
- Tokenizer: `allenai/OLMo-2-0425-1B-SFT` at revision `0d85a3d037876ce6ac7d4311d994400fc66ac27f`, with the chat template applied as TRL 1.14 does: to the prompt alone with the generation prompt, to prompt + chosen and to prompt + rejected.
- A pair is truncated when the longer of its two sequences has more than `max_length` tokens, and dropped when its prompt alone has `max_length` tokens or more.

### Result

**Check 1.**

| Sequence | Median | 90% | 95% | 99% | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| prompt | 29 | 158 | 378 | 1,378 | 2,597 |
| prompt + chosen | 391 | 790 | 1,012 | 1,959 | 4,778 |
| prompt + rejected | 293 | 712 | 938 | 1,937 | 7,502 |
| longer of the two | 444 | 821 | 1,068 | 2,100 | 7,502 |

- **Most prompts are short** (median 29 tokens), but the longest 1% have more than 1,378 tokens, such as pasted documents or code.
- **The chosen answer tends to be longer than the rejected one** (median 391 against 293 tokens with the prompt). This fits the known preference of annotators for longer answers, so DPO may teach the model to write longer answers. <ins>Answer length is worth logging in the evaluation.</ins>

**Check 2.** Longer sequence of each pair:

| Tokens | Pairs | Share |
| --- | ---: | ---: |
| 0–255 | 2,564 | 24.5% |
| 256–511 | 3,611 | 34.5% |
| 512–767 | 2,967 | 28.4% |
| 768–1,023 | 746 | 7.1% |
| 1,024–1,535 | 352 | 3.4% |
| 1,536–2,047 | 106 | 1.0% |
| 2,048–3,071 | 89 | 0.9% |
| 3,072–4,095 | 16 | 0.2% |
| 4,096+ | 10 | 0.1% |

**Check 3.**

| `max_length` | Truncated | Strong | Moderate | Split | Dropped |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 4,270 (40.8%) | 42.5% | 41.1% | 37.1% | 397 (3.8%) |
| 1,024 | 573 (5.5%) | 5.9% | 5.5% | 4.6% | 162 (1.5%) |
| 1,536 | 220 (2.1%) | 2.2% | 2.2% | 1.7% | 72 (0.7%) |
| 2,048 | 115 (1.1%) | 1.2% | 1.1% | 0.9% | 31 (0.3%) |
| 4,096 | 10 (0.1%) | 0.2% | 0.1% | 0.0% | 0 |

- **1,024 truncates 5.5% of the pairs and drops 1.5%.** The dropped pairs are among the truncated ones: their prompt alone fills `max_length`, so no answer token is left to train on.
- **Truncation hits the three buckets about equally,** with split pairs slightly less often, so no `max_length` changes the balance of the buckets much.
- **2,048 truncates 1.1% and drops 0.3%,** and 4,096 truncates only the 10 longest pairs.

### Decision

Use `max_length` 2,048 for training and for the evaluation of every condition, if it fits in memory.

- **1,024 loses too much.** It cuts the end of 573 pairs, where the two answers may differ most, and removes 162 pairs from training.
- **2,048 keeps almost everything.** Only 115 pairs (1.1%) are cut and 31 (0.3%) dropped. 4,096 would save 105 more pairs at a much larger cost in memory.
- **The cost is in memory, not in time.** TRL pads each batch to its longest sequence, so only the few batches with a long pair get longer. The smoke test peaked at 15.24 GiB with batch size 2 and `max_length` 1,024. The model and optimizer take a fixed share of that, so 2,048 should stay below double. Measure the peak in plan step 2, and fall back to 1,536 or to batch size 1 with gradient accumulation if it doesn't fit.
- **Use the same `max_length` in the evaluation.** The margins h on the test split are computed by our own code, and cutting the sequences in the same way as training keeps every condition compared on the same text.

## Position of the answers

- **Script:** [`count_position_wins.py`](count_position_wins.py)
- **Output:** printed to the terminal and copied into the result below.

### Question

In the [MultiPref annotations](#multipref-annotations) experiment, B had the majority in 67.7% of the pairs with a majority. Either the annotators favour the answer shown in position B or the stronger models were more often placed in position B. In a pair where both answers come from the same model, model strength can't favour either position, so B's share there tells the two apart:

1. What is B's share of the majorities over all pairs with a majority?
2. What is it over the pairs where both answers come from the same model?
3. How often does each model appear in position A and in position B?

### Setup

- Dataset: `nelsonfilipecosta/multipref-soft-labels` at revision `96819a1cda4e6391735e4078f4fe8e15afe3adec`, all three splits (10,461 pairs).
- The models in positions A and B are recovered from `chosen_model`, `rejected_model` and `chosen_is_a`.
- B has the majority when the pair isn't split (p ≠ 0.5) and the chosen answer isn't A.
- Each share comes with a 95% Wilson confidence interval, which stays between 0 and 1 and works for small samples.

### Result

**Check 1.** B has the majority in 5,842 of the 8,625 pairs with a majority (67.7%, 95% CI 66.7%–68.7%), as in the MultiPref annotations experiment.

**Check 2.** 2,919 of the 10,461 pairs (27.9%) compare a model with itself.

| Pairs | Pairs with a majority | B has the majority | 95% CI |
| --- | ---: | ---: | ---: |
| same model | 2,273 | 1,259 (55.4%) | 53.3%–57.4% |
| different models | 6,352 | 4,583 (72.2%) | 71.0%–73.2% |

- **Annotators favour position B a little.** With the same model in both positions, B still has 55.4% of the majorities and the interval excludes 50%.
- **Most of B's advantage comes from the models.** The 55.4% is far below the 67.7% overall and the interval excludes it.
- **Pairs of the same model are split more often:** 646 of 2,919 (22.1%) against 1,190 of 7,542 (15.8%) for pairs of different models. Annotators disagree more when the two answers are alike.

**Check 3.** Appearances of each model in position A and B over all pairs:

| Model | A | B | Share in B |
| --- | ---: | ---: | ---: |
| `allenai/tulu-2-7b` | 3,059 | 517 | 14.5% |
| `gpt-4-turbo-2024-04-09` | 1,517 | 2,001 | 56.9% |
| `allenai/tulu-2-70b` | 2,517 | 998 | 28.4% |
| `meta-llama/Meta-Llama-3-70B-Instruct` | 476 | 3,010 | 86.3% |
| `meta-llama/Llama-2-70b-chat-hf` | 971 | 2,481 | 71.9% |
| `gpt-3.5-turbo-0125` | 1,921 | 1,454 | 43.1% |

- **The models were not placed at random.** Llama 3 70B is in position B 86.3% of the time and Tulu 2 7B only 14.5%, while each model appears in about a third of the pairs.
- **The placement favours the stronger models in B.** The smallest model, Tulu 2 7B, is mostly in A, and one of the strongest, Llama 3 70B, mostly in B. This fits the 72.2% for B in pairs of different models.

### Decision

Keep the data as it is and note both findings in the write-up.

- **Training doesn't see the positions.** Every pair is turned into chosen and rejected answers before training, so DPO can't learn to prefer a position.
- **The small position bias is part of the label noise.** In pairs of the same model, B gains about 5 points over the 50% expected without bias, so some majorities partly reflect where an answer was shown. The soft labels carry this as disagreement, like any other source of it.
- **"B wins 67.7%" is not an annotator bias.** Most of it comes from the placement of the models, so it should not be quoted as one.
- **Same-model pairs are a natural group of hard pairs.** They are split more often, so they are worth a look in the evaluation of split pairs.
