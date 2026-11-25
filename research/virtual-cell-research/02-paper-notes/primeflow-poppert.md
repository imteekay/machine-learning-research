# PRiMeFlow and PopPert: flow matching and population-level distributions for heterogeneous perturbation responses

> PRiMeFlow: Yan Z, Wu Y, et al., Stark R. *PRiMeFlow: Capturing Complex Expression Heterogeneity in Perturbation Response Modelling*. arXiv (2026) · arXiv: 2604.13986 · Code: not found. PopPert: Wang H, Qi J, Feng H, Lai B. *PopPert: Population-level Joint-Distribution Modeling for Single-Cell Perturbation Prediction*. arXiv (2026) · arXiv: 2609.01357 · Code: [whd1125/PopPert](https://github.com/whd1125/PopPert)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Both notes are **abstract-only** (arXiv abstract pages).

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: PRiMeFlow is a flow-matching model in gene-expression space (U-Net velocity field) for genetic and small-molecule perturbations, scaled to a multi-dataset atlas with pretrain/finetune. PopPert instead predicts shifts in the parameters of a population-level joint distribution (low-rank Gaussian copula) from a control population. **(paper)**
- **Why it matters for the virtual cell**: two opposite answers to "what is the prediction target": a per-cell generative map vs a population distribution. This is the heterogeneity question behind E4.3. **(my take)**
- **Main claim**: PRiMeFlow reports strong results on PerturBench and on H1 hESC from the ARC Virtual Cell Challenge; PopPert reports superior DE recovery, effect estimation and distribution matching, with lower sensitivity to single-cell noise. **(paper, abstract-level, no numbers seen)**

## 2. Problem it solves

Perturbed cells are never paired with their own control (scRNA-seq is destructive), and responses are heterogeneous; mean-shift models discard that. PopPert argues cell-to-cell correspondence assumptions are a mismatch. **(paper)**

## 3. Method

PRiMeFlow: flow matching directly on expression, U-Net velocity, multi-dataset pretraining then finetuning, ablations on design choices. PopPert: parameterizes joint gene distribution with a low-rank Gaussian copula; maps (control population, condition) to parameter shifts; can sample synthetic perturbed cells. Further detail abstract-only.

## 4. Data

PerturBench and ARC Virtual Cell Challenge H1 hESC (PRiMeFlow); PopPert: "multiple benchmarks" of genetic and chemical perturbations, names not retrieved. Atlas composition abstract-only.

## 5. Evaluation and results

Abstract-only: no metrics or baseline strength retrieved. Need to check whether mean/linear baselines were included.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): a Gaussian copula may miss multimodal responder/non-responder structure; flow matching in 2000+ dim expression space is costly and metric-sensitive; neither is shown (from abstracts) to beat a linear baseline OOD.
- What would change my mind: PopPert beating a flow model on a bimodal-response dataset, or PRiMeFlow's atlas pretraining helping on a held-out cell line vs training from scratch.

## 7. Relevance to my research

- Proposal link: the [research proposal](../00-plan/research-proposal.md) H1 needs a distribution-aware target; these are two candidate heads on top of an RNA+ATAC encoder (E4.1, E4.3).
- Hypotheses worth testing:
  - E4.3: on a dataset with a known responder/non-responder split, a flow model recovers the responder fraction within 5 points while a copula population model does not (falsified if the copula matches).
  - Population-level parameter prediction is more robust than per-cell prediction when cells per condition < 50 (falsified if the gap does not grow as cells are subsampled).

## 8. Related work

CellFlow, STATE ([state-arc](state-arc.md)), scGen ([scgen](scgen.md)), PerturBench.

## Resources

- [PRiMeFlow arXiv](https://arxiv.org/abs/2604.13986) · [PopPert arXiv](https://arxiv.org/abs/2609.01357) · [PopPert code](https://github.com/whd1125/PopPert)
