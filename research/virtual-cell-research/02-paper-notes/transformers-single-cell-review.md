# Transformers in single-cell omics: review of how transformers are adapted to gene tokens and what they do well

> Szałata A, Hrovatin K, Becker S, Tejada-Lapuerta A, Cui H, Wang B, Theis FJ. *Transformers in single-cell omics: a review and new perspectives*. Nature Methods (2024). DOI: [10.1038/s41592-024-02353-z](https://doi.org/10.1038/s41592-024-02353-z) · Code: none

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 0](../00-plan/roadmap.md#phase-0-foundations), [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: describes the transformer architecture and its single-cell adaptations, reviews applications in single-cell analysis, and discusses limits and future directions. **(paper)**
- **Why it matters for the virtual cell**: the bridge from generic transformer knowledge to scFM design choices (how a cell becomes a sequence of tokens), which decides what a model can and cannot learn. **(my take)**
- **Main claim**: transformers became the default for FMs elsewhere because they generalize across large, heterogeneous data; whether the same shift happens in single-cell biology is open, and the review lays out the technical challenges. **(paper)**

## Research summary

- **Context (already known)**: Transformers became the default for foundation models in other fields. **(paper)**
- **Gap (not known)**: Whether and how this shift will happen in single-cell biology given technical challenges. **(paper)**
- **Hypothesis / what they try to determine**: To describe transformer adaptations for single-cell data and assess limits and directions. **(paper)**
- **Approach (how they investigated)**: Review of architecture, applications, and limitations (abstract-only; paywalled). **(paper)**
- **Results (what they found)**: Lays out technical challenges; no new data or numbers. **(paper)**
- **Implications (what it means)**: Tokenization and set structure are likely the key design choices, more than size. **(my take)**

## 2. Problem it solves

Single-cell atlases are large, but current models do not fully use them. Readers need a structured view of how transformers have been adapted and where they fall short. **(paper)**

## 3. Method

Abstract-only (Nature paywall). Known from the abstract: (1) transformer architecture and single-cell adaptations, (2) review of applications, (3) limitations and outlook. **(paper)** Background I bring **(my take)**: the main design axis is tokenization: rank order (Geneformer), binned expression values (scBERT, scGPT), or continuous value embeddings; plus how gene identity is embedded and how the set (unordered) nature of genes is handled.

## 4. Data

No new dataset.

## 5. Evaluation and results

Review; abstract-only for any comparative tables.

## 6. Critical analysis

- Limitations (paper): abstract-only.
- Weaknesses I see (my take): gene expression profiles are sets, not sequences. Attention over ~2k–20k genes is costly, and positional structure from language does not transfer. These choices may matter more than model size.
- What would change my mind: a controlled study where tokenization changes but data and compute are fixed, showing little effect on downstream tasks.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: the Master's plan asks for a transformer mapping RNA and ATAC into a shared latent space. Adding ATAC means choosing a token for peaks (peak, gene-linked activity score, or motif), which this review's framing helps compare.
- Hypotheses worth testing:
  - **H-TOK-1**: with data and compute fixed, rank tokenization vs binned values vs continuous embeddings change E2.1 (cell-type kNN accuracy vs PCA) by less than the CI. If true, tokenization is not the bottleneck.
  - **H-TOK-2**: for ATAC, gene-level activity tokens give smaller E4.1 gains than peak-level tokens, because peak-level keeps distal regulatory information. Links to E3.1.

## 8. Related work

- [Multimodal FMs perspective](multimodal-foundation-models-cell-biology.md) (overlapping authors), [scBERT](scbert.md), [Geneformer](geneformer.md), [scGPT](scgpt.md), [Nicheformer](nicheformer.md).

## Resources

- DOI: https://doi.org/10.1038/s41592-024-02353-z
