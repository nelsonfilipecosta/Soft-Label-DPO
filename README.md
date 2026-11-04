# Soft-Label-DPO

Does a model trained with DPO on majority labels become more confident than its annotators were?

## Motivation

Preference datasets usually keep only the majority label of several annotators. When the annotators are divided, this label still tells the model that one answer is clearly better. About 39% of MultiPref pairs show diverging preferences, yet reward models trained on majority labels predict decisive preferences on them ([Zhang et al., 2025](https://arxiv.org/abs/2410.14632)). That work and [Halpern et al. (2025)](https://arxiv.org/abs/2506.06298) study reward models, while many models are now trained directly on preference pairs with DPO. Soft preference labels have been used with DPO before ([Furuta et al., 2024](https://arxiv.org/abs/2409.06691)), but simulated from AI feedback rather than taken from real human votes.

## Approach

I train OLMo 2 1B SFT (`allenai/OLMo-2-0425-1B-SFT`) on MultiPref, where each pair has four votes from two crowdworkers and two experts. Three conditions use the same pairs, the same settings and three seeds each. They differ only in the target probability with which the model should learn to prefer the chosen answer:

- **Majority:** the majority label (standard DPO).
- **Soft:** the share of annotators who preferred the chosen answer.
- **Constant control:** the same average vote share on every pair.

The constant control is there to separate the information in per-pair disagreement from plain label smoothing. Soft targets make training gentler, so the soft model has only learned where people disagree if it beats this control.

On held-out pairs, the main metric is the rank correlation between the decisiveness of the model (how far its implied preference probability is from 50/50) and the decisiveness of the annotators (how far their vote share is from 50/50). I also report calibration error by level of agreement and the AUROC for detecting diverging pairs from [Zhang et al. (2025)](https://arxiv.org/abs/2410.14632). Finally, an LLM judge checks whether answers to the most and least contested prompts acknowledge reasonable alternatives.

**Project plan** (private working doc): [MultiPref DPO Project Plan](https://claude.ai/artifact/KDuz7HwGZUhS6bQrEkUWjZ)

## Smoke tests

Two short DPO runs check the training pipeline end to end before any real experiment.

**Local** ([configs/smoke_test_local.yaml](configs/smoke_test_local.yaml)): 3 steps with a tiny random model on the Mac CPU. It only checks that the run finishes without errors. The Mac GPU (MPS) is not used because it computes wrong losses and gradients once batches are long.

**Rented GPU** ([configs/smoke_test.yaml](configs/smoke_test.yaml)): 10 steps of standard DPO on OLMo 2 1B SFT (`allenai/OLMo-2-0425-1B-SFT`), with 20 pairs from `trl-lib/ultrafeedback_binarized`, batch size 2, `max_length` 1024, bf16 and gradient checkpointing. Run on RunPod with an NVIDIA RTX 4000 Ada Generation (20 GB) on 2026-11-04.

| Metric | Value |
| --- | --- |
| Loss at step 1 | 0.6931 (= ln 2, with margins of 0, as expected before any update) |
| Loss over the 10 steps | 0.647 to 0.719 (mean 0.6896) |
| Gradient norm before clipping | 102 to 488 |
| Peak GPU memory | 15.24 GiB |
| Speed | 0.736 steps/s (1.47 pairs/s) |
| Training time | 13.6 s |
| Pod setup time ([scripts/setup_pod.sh](scripts/setup_pod.sh)) | 294 s |
| Cost | $0.08 |

- **Memory is tight at batch size 2.** At step 4, one 1.64 GB allocation failed and PyTorch retried it after freeing cached memory, so the run continued. That allocation is the float32 logits for 4 sequences (2 chosen, 2 rejected) × 1024 tokens × 100,352 vocabulary entries. Real runs need a larger GPU or gradient accumulation.
- **Time estimate for real runs.** At this speed, one epoch over about 7,500 MultiPref training pairs takes about 85 minutes on this GPU, on top of about 5 minutes to set up a new pod.
