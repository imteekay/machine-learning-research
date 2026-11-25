# CellRFT: reinforcement fine-tuning for single-cell perturbation modeling

> Yan J, Liu L, Guo H, ..., Wang Y. *CellRFT: Reinforcement Fine-Tuning for Single-Cell Perturbation Modeling*. arXiv (2026) · arXiv: 2609.19970 · Code: "will be made available"

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. This note is **abstract-only**.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: policy-gradient fine-tuning of pretrained perturbation models using non-differentiable biological evaluations of generated cell populations as rewards, with hierarchical reward aggregation. **(paper)**
- **Why it matters for the virtual cell**: training losses (MSE, MMD) mismatch the biological metrics we care about; RL can optimize the metric directly. **(my take)**
- **Main claim**: works across pretrained models, improves perturbation prediction, and complementary rewards improve criteria beyond those directly optimized; single rewards can trade off against others. **(paper, no numbers seen)**

## 2. Problem it solves

Surrogate losses misalign with biological evaluation criteria. **(paper)**

## 3. Method

Abstract-only: policy gradient, multiple rewards, hierarchical aggregation.

## 4. Data

Not retrieved.

## 5. Evaluation and results

Abstract-only; no metrics or baselines.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): reward hacking on metrics that are themselves flawed (e.g. Pearson delta driven by systematic variation); RL on top of a base model adds variance.
- What would change my mind: gains on a held-out metric not in the reward set under strict OOD splits.

## 7. Relevance to my research

- Proposal link: a cheap post-hoc step on the [research proposal](../00-plan/research-proposal.md) H1 model to optimize DE-AUPRC or distributional metrics.
- Hypotheses worth testing:
  - E4.3: RL with a responder-fraction reward improves responder recovery without lowering Pearson delta (falsified if it degrades >2 points).
  - E4.4: reward-optimized models do not improve against a mean baseline when the reward is systematic-variation-controlled.

## 8. Related work

PRiMeFlow ([primeflow-poppert](primeflow-poppert.md)), scDEBART ([scdebart](scdebart.md)), STATE ([state-arc](state-arc.md)).

## Resources

- [arXiv](https://arxiv.org/abs/2609.19970)
