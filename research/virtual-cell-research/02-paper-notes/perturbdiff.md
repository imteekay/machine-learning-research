# PerturbDiff: diffusion over distributions in Hilbert space

> Yuan X, Liu X, Zhang YS, Zhang Z, Guo H, Tang J. *PerturbDiff: Functional Diffusion for Single-Cell Perturbation Modeling*. arXiv:[2602.19685](https://arxiv.org/abs/2602.19685) (2026) · Code: [DeepGraphLearning/PerturbDiff](https://github.com/DeepGraphLearning/PerturbDiff) (announced)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. This note is **abstract-only**: the full text was not retrieved (paywall / rate limit), so no claim below comes from the methods or results sections.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: embed whole cell distributions as points in a Hilbert space and run diffusion over those points, so the model generates a *distribution of response distributions*. **(paper)**
- **Why it matters for the virtual cell**: explicitly models hidden-factor variability (microenvironment, batch) as uncertainty over response distributions — the latent context the thesis wants to make observable. **(my take)**
- **Main claim**: state-of-the-art response prediction and better generalization to unseen perturbations. **(paper)**

## 2. Problem it solves

Existing models assume one fixed response distribution per (context, perturbation); hidden factors make it a manifold of distributions. **(paper)**

## 3. Method

Distribution embedding (kernel-mean-like) into Hilbert space; diffusion process over those functions. Details abstract-only. **(paper)**

## 4. Data

'established datasets' (abstract-only).

## 5. Evaluation and results

no numbers (abstract-only).

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): hidden-factor variance may be mostly batch; measured context (ATAC) could explain part of it, making the functional diffusion less necessary. **(my take)**
- What would change my mind: showing the learned variance is biologically structured, not batch.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: connects to H1: if ATAC explains hidden variation, PerturbDiff's sample spread should shrink when conditioned on it. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: conditioning on ATAC reduces PerturbDiff's predictive variance across replicate-like contexts; falsified if unchanged. (2) *H* (E4.3): it beats scDFM on E-distance for heterogeneous responses.

## 8. Related work

scDFM ([note](scdfm.md)), Squidiff ([note](squidiff.md)), CellOT.

## Resources

- Project: https://katarinayuan.github.io/PerturbDiff-ProjectPage/
