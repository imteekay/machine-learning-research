# scDEBART: in silico perturbation via large-scale differential expression learning

> *scDEBART: Predicting in silico Single-Cell Perturbation Responses via Large-Scale Differential Expression Learning*. ICML (2026) · [poster](https://icml.cc/virtual/2026/poster/61521) · [OpenReview](https://openreview.net/forum?id=pJyidZg93y) · Code: unknown

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. This note is **abstract-only** (OpenReview blocked; ICML poster summary via search).

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: learns from differential-expression signatures across a very large number of cell comparisons, with a statistical denoising step before training. **(paper, poster summary)**
- **Why it matters for the virtual cell**: it optimizes the thing biologists use (which genes change) instead of raw expression MSE. **(my take)**
- **Main claim**: on five large genetic perturbation experiments, recovers truly affected genes 4-7x more than existing models and can infer which gene was perturbed from the expression change. **(paper, unverified)**

## Research summary

- **Context (already known)**: Perturbation models usually optimize expression-level losses, which unchanged genes dominate; DE signatures are what biologists use. **(my take)**
- **Gap (not known)**: Whether learning DE signatures across a very large number of comparisons, with statistical denoising, recovers truly affected genes better. **(paper, poster summary)**
- **Hypothesis / what they try to determine**: A DE-learning objective gives better recovery of affected genes and allows inferring the perturbed gene. **(paper)**
- **Approach (how they investigated)**: Train on DE signatures from many cell comparisons after statistical cleaning; evaluate on five large genetic perturbation experiments. Architecture not available in abstract. **(paper, abstract-level)**
- **Results (what they found)**: Recovers truly affected genes 4-7x more than existing models and infers the perturbed gene; unverified. **(paper)**
- **Implications (what it means)**: If robust to mean/linear DE baselines it supports DE-based losses; circularity and target-gene shortcuts are open concerns. **(my take)**

## 2. Problem it solves

Expression-level losses are dominated by unchanged genes, so models miss DE genes. **(my take, consistent with the AUPRC literature)**

## 3. Method

Abstract-only: DE-learning over many comparisons plus statistical cleaning. Architecture not retrieved.

## 4. Data

Five large genetic perturbation experiments (names not retrieved).

## 5. Evaluation and results

Abstract-only: DE recovery (probably AUPRC-like) vs "existing models"; baseline strength unknown.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): DE labels derived from the same data can be circular; the reverse task (infer the perturbed gene) can be solved by shortcuts such as target-gene expression drop.
- What would change my mind: a mean/linear DE-ranking baseline matching the 4-7x gain.

## 7. Relevance to my research

- Proposal link: gives a DE-based metric and loss for the [research proposal](../00-plan/research-proposal.md) H1 evaluation.
- Hypotheses worth testing:
  - E4.4: perturbation inference accuracy falls to near chance when the target gene is masked from the input (falsified if it stays high).
  - E0/E4: a DE-weighted loss in the RNA+ATAC model raises DE-AUPRC more than MSE-trained models (falsified if gain < noise across seeds).

## 8. Related work

AUC-PR metric paper, PertAdapt adaptive DE loss ([pertadapt-closing-the-loop](pertadapt-closing-the-loop.md)), GEARS.

## Resources

- [ICML poster](https://icml.cc/virtual/2026/poster/61521) · [OpenReview](https://openreview.net/forum?id=pJyidZg93y)
