# Fundamental limitations of scFMs: noise-robust cell typing vs a Seurat baseline

> Authors not confirmed (abstract-only). *Fundamental Limitations of Foundation Models in Single-Cell Transcriptomics*. bioRxiv (2025). DOI: [10.1101/2025.06.26.661767](https://www.biorxiv.org/content/10.1101/2025.06.26.661767v1) · Code: not checked

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: compares scGPT, SCMAMBA-2 and Geneformer with a statistical baseline (Seurat v5) for cell-type classification under added Gaussian noise. **(paper, abstract-level)**
- **Why it matters for the virtual cell**: it argues that the gap is *structural*, from missing biological context, not something more data will fix. That is the core claim of the research proposal. **(my take)**
- **Main claim**: the FMs underperform the statistical baseline, and the authors attribute this to the lack of biological context in the models. **(paper)**

## 2. Problem it solves

It asks whether scFM failures are fixable by scale or come from what the models can represent. **(paper)**

## 3. Method

Abstract-only: cell-type classification with increasing Gaussian noise. Architectures, fine-tuning protocol and noise levels were not checked. The bioRxiv full text returned HTTP 429.

## 4. Data

Abstract-only. Datasets not confirmed.

## 5. Evaluation and results

Abstract-only. The direction is that the FMs score below Seurat v5. I could not get the numbers. **(paper)**

## 6. Critical analysis

- Limitations (paper): not retrieved (abstract-only).
- Weaknesses I see (my take): Gaussian noise on normalized expression is not a realistic single-cell perturbation. Count noise, dropout and depth changes are. "Missing biological context" as an explanation needs a positive control: add context and show the gap closes.
- What would change my mind: a context-augmented FM (e.g., with ATAC or a gene-graph prior) that closes the noise-robustness gap where a bigger expression-only model does not.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: it directly motivates the "context, not scale" hypothesis and Master's H1 (E4.1).
- Hypotheses worth testing:
  - **Noise-robustness sub-experiment for E2.1**: corrupt counts by binomial thinning (depth 100% → 10%) instead of Gaussian noise. H: FM annotation accuracy degrades *faster* than Seurat/PCA. Null: equal slopes.
  - **E3.1 link**: H: adding ATAC to a classifier flattens the degradation curve under RNA thinning more than doubling the RNA pretraining data does.

## 8. Related work

[Simple models vs scFMs](simple-models-beat-scfms.md), Kedzierska 2025 (zero-shot limits, Genome Biology), [context-dependent utility](scfm-context-dependent-utility.md).

## Resources

- bioRxiv: https://www.biorxiv.org/content/10.1101/2025.06.26.661767v1
