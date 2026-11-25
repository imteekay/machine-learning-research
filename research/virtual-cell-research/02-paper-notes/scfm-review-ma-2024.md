# Foundation models in single-cell omics (NRMCB Comment): a biologist-friendly overview of progress, limits and best practice

> Ma Q, Jiang Y, Cheng H, Xu D. *Harnessing the deep learning power of foundation models in single-cell omics*. Nature Reviews Molecular Cell Biology (2024), Comment. DOI: [10.1038/s41580-024-00756-6](https://doi.org/10.1038/s41580-024-00756-6) · Code: none

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 0](../00-plan/roadmap.md#phase-0-foundations), [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a short Comment reviewing progress, limitations and best practices for using foundation models (FMs) on single-cell omics. **(paper)**
- **Why it matters for the virtual cell**: an entry-level orientation before reading scBERT, Geneformer and scGPT themselves. **(my take)**
- **Main claim**: FMs are promising for single-cell analysis, but several challenges still need work. **(paper)**

## 2. Problem it solves

Biologists need a short guide to when and how to use scFMs. **(paper, from summary)**

## 3. Method

Abstract-only: I could only access a one-paragraph summary (Nature paywall; no abstract in Europe PMC). The specific challenges and best practices listed in the Comment were not checked.

## 4. Data

No dataset. Abstract-only.

## 5. Evaluation and results

No experiments. Abstract-only.

## 6. Critical analysis

- Limitations (paper): abstract-only.
- Weaknesses I see (my take): a 2024 Comment predates most of the negative benchmark results (scFMs vs PCA/linear baselines for zero-shot and perturbation tasks). Read its optimism with that in mind.
- What would change my mind: n/a (no claims checked).

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: background for the Phase 2 rung of the hypothesis ladder ("pretrained scFM embeddings carry information beyond PCA").
- Hypotheses worth testing: none new; the open question it frames is the one E2.1 and E2.2 already test (do scFM embeddings beat PCA zero-shot, and does fine-tuning close the gap at small data sizes?).

## 8. Related work

- [Transformers in single-cell omics](transformers-single-cell-review.md) (more technical), [scBERT](scbert.md), [Geneformer](geneformer.md), [scGPT](scgpt.md).

## Resources

- DOI: https://doi.org/10.1038/s41580-024-00756-6
