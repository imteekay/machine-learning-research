# PerturbDiff: diffusion over distributions in Hilbert space

> Yuan X, Liu X, Zhang YS, Zhang Z, Guo H, Tang J. *PerturbDiff: Functional Diffusion for Single-Cell Perturbation Modeling*. arXiv:[2602.19685](https://arxiv.org/abs/2602.19685) (2026) · Code: [DeepGraphLearning/PerturbDiff](https://github.com/DeepGraphLearning/PerturbDiff) (announced)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Updated from the arXiv HTML full text (main text; appendix tables C.3 not read).

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read (arXiv HTML, first ~100k of 215k characters; appendix tables not read)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: embed whole cell distributions as points in a Hilbert space and run diffusion over those points, so the model generates a *distribution of response distributions*. **(paper)**
- **Why it matters for the virtual cell**: explicitly models hidden-factor variability (microenvironment, batch) as uncertainty over response distributions — the latent context the thesis wants to make observable. **(my take)**
- **Main claim**: state-of-the-art response prediction and better generalization to unseen perturbations. **(paper)**

## Research summary

- **Context (already known)**: Perturbation models assume one fixed response distribution per context and perturbation, and mean, linear, CPA, STATE, CellFlow and Squidiff are common comparators. **(paper)**
- **Gap (not known)**: Whether hidden factors (microenvironment, batch) are better modeled as uncertainty over response distributions is not established. **(paper)**
- **Hypothesis / what they try to determine**: Treat the perturbed population as a random variable and model a distribution of distributions with diffusion in a Hilbert space. **(paper)**
- **Approach (how they investigated)**: Kernel mean embeddings of cell batches, DDPM-style diffusion with a hybrid MMD+MSE loss and an MM-DiT, optional pretraining on ~61M CellxGene cells; tested on PBMC, Tahoe100M and Replogle with STATE splits. **(paper)**
- **Results (what they found)**: Outperforms baselines on nearly all metrics on PBMC and Tahoe100M, second on Replogle (trails STATE); finetuning helps most in low-data regimes; largest model overfits. Appendix tables not read. **(paper)**
- **Implications (what it means)**: Distribution-level modeling helps DE metrics, but whether the learned variance is biological rather than batch is untested. **(my take)**

## 2. Problem it solves

Existing models assume one fixed response distribution per (context, perturbation); hidden factors make it a manifold of distributions. **(paper)**

## 3. Method

Kernel mean embedding of cell batches into an RKHS; DDPM-style forward process over embeddings; Gaussian RKHS noise approximated by per-cell Gaussian noise (first-order linearization); hybrid MMD + MSE loss; MM-DiT with joint attention over control and perturbed streams and AdaLN-Zero conditioning; deterministic DDIM sampling in cell space; optional pretraining on unperturbed cells (~61M CellxGene). Variants: From Scratch, Finetuned. **(paper)**

## 4. Data

PBMC cytokines (9.7M cells, 90 perturbations), Tahoe100M (101.2M cells, 1,137 drugs), Replogle CRISPRi (0.6M cells, 2,023 perturbations, ~300 cells each); top 2,000 HVGs; STATE splits; CellxGene 60.9M cells for pretraining. **(paper)**

## 5. Evaluation and results

Baselines: Mean, Linear, CPA, STATE, CellFlow, Squidiff; metrics via Cell-Eval (R2, Pearson delta, PDS, DE overlap/precision, AUROC, AUPRC and others). From Scratch beats all on nearly all metrics on PBMC and Tahoe100M (win rate vs STATE on DE metrics reported above 96-100%) and is second on Replogle; Finetuned matches STATE on Replogle and mainly improves DE metrics; with 1-5% PBMC data Finetuned clearly beats From Scratch; Linear collapses on Replogle (PDS ~0.5); the 114M model is best and 239M overfits; removing MMD hurts. Exact table values not read. **(paper)**

## 6. Critical analysis

- Limitations (paper): no dedicated section; finetuning mainly helps DE metrics not average accuracy, trails STATE on low-data Replogle, linearized RKHS noise approximation, non-monotonic scaling, code and data announced not released.
- Weaknesses I see (my take): hidden-factor variance may be mostly batch; measured context (ATAC) could explain part of it, making the functional diffusion less necessary. **(my take)**
- What would change my mind: showing the learned variance is biologically structured, not batch.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: connects to H1: if ATAC explains hidden variation, PerturbDiff's sample spread should shrink when conditioned on it. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: conditioning on ATAC reduces PerturbDiff's predictive variance across replicate-like contexts; falsified if unchanged. (2) *H* (E4.3): it beats scDFM on E-distance for heterogeneous responses.

## 8. Related work

scDFM ([note](scdfm.md)), Squidiff ([note](squidiff.md)), CellOT.

## Resources

- Project: https://katarinayuan.github.io/PerturbDiff-ProjectPage/
