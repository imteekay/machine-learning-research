# Tiny Recursive Models for spatial omics: recursion as a cheap path to global tissue reasoning

> Authors not retrieved. *Reasoning Across Space: Tiny Recursive Models for Spatial Omics*. ICML (2026). OpenReview: [vtTox3sHaS](https://openreview.net/pdf?id=vtTox3sHaS) · Code: not found
>
> OpenReview blocked automated fetch; built from the ICML virtual-site abstract. **Abstract-only**.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: applies Tiny Recursive Models (small networks iterated many times on their own latent state) to spatial omics, to reason over whole tissue samples instead of single-pass local patches, without test-time training **(paper)**.
- **Why it matters for the virtual cell**: a perturbation's ripple through tissue is an iterative propagation; recursion is a structural match, and small models are trainable on a student budget.
- **Main claim**: beats several slice-specific unsupervised methods at domain identification; SOTA among supervised methods for gene expression prediction; beats parameter-matched supervised baselines; recursion enables test-time scaling/adaptive compute; domain ID generalizes only partially across tissues **(paper)**.

## 2. Problem it solves

Current spatial methods reason locally in one pass; large FMs are expensive **(paper)**.

## 3. Method

TRM-style recursion over tissue representations; details abstract-only.

## 4. Data

Abstract-only.

## 5. Evaluation and results

See TL;DR; numbers abstract-only. Authors honestly report partial cross-tissue generalization.

## 6. Critical analysis

- Limitations (paper): partial generalization of domain identification across tissue types.
- Weaknesses I see (my take): "gene expression prediction" may be histology->expression or imputation; which one matters for comparison (abstract-only).
- What would change my mind: recursion depth showing no gain over a deeper non-recursive model of equal FLOPs.

## 7. Relevance to my research

- **Hypotheses**: PhD H — iterating a local update k steps mimics signal propagation k cell-hops away; recursion steps could be interpreted as time **(my take)**.
- Hypotheses worth testing:
  - **H-a (E5.2)**: in Perturb-map, prediction error at distance d from the KO patch drops as recursion depth increases up to ~d hops, then plateaus; falsified if depth has no distance-dependent effect.
  - **H-b**: a small recursive model matches a large spatial FM (e.g. [SToFM](stofm.md)) on E5.1 niche tasks at <10% parameters.

## 8. Related work

Tiny Recursive Model / HRM (ARC reasoning), [SToFM](stofm.md), [CellWorld](cellworld.md).

## Resources

- OpenReview PDF: https://openreview.net/pdf?id=vtTox3sHaS
- ICML page: https://icml.cc/virtual/2026/70191
