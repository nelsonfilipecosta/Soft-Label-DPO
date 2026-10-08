# Experiments

Each experiment is a one-off check. Each entry gives the question, the setup, the result and the decision it led to.

| Experiment | Question | Decision |
| --- | --- | --- |
| [BOS token in the chat template](#bos-token-in-the-chat-template) | Does the leading `<\|endoftext\|>` in the chat template affect the model's answers? | Keep the template as shipped and use it everywhere. |

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

- Model: `allenai/OLMo-2-0425-1B-SFT`.
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
