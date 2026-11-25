# Zero-shot evaluation of scGPT and Geneformer vs simple baselines

> Kedzierska KZ, Crawford L, Amini AP, Lu AX. *Zero-shot evaluation reveals limitations of single-cell foundation models*. Genome Biology (2025). DOI: [10.1186/s13059-025-03574-x](https://doi.org/10.1186/s13059-025-03574-x) · PMC12007350 · Code: [microsoft/zero-shot-scfoundation](https://github.com/microsoft/zero-shot-scfoundation)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Only the abstract or secondary sources were read for this note (**abstract-only**); details beyond those are marked as missing rather than guessed.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: zero-shot benchmark of scGPT and Geneformer embeddings for cell-type clustering and batch integration. **(paper)**
- **Why it matters for the virtual cell**: it is the central critique behind **E2.1**: frozen FM embeddings may add nothing over HVG + standard methods. **(my take)**
- **Main claim**: both models are worse than HVG selection, Harmony and scVI on cell-type clustering (AvgBio); Geneformer also underperforms on batch integration. **(paper)**

## 2. Problem it solves

Exploratory analyses use embeddings without labels, so fine-tuning benchmarks overstate FM usefulness. **(paper)**

## 3. Method

Compare zero-shot embeddings from scGPT variants and Geneformer to HVG, Harmony, scVI on several datasets using scIB-style metrics (AvgBio, batch scores). **(paper)**

## 4. Data

Multiple public datasets including Pancreas and PBMC 12k. **(paper)**

## 5. Evaluation and results

- HVG / Harmony / scVI outperform both FMs on AvgBio clustering. **(paper)**
- Geneformer underperforms scGPT, Harmony and scVI on batch integration; Harmony and scVI integrate Pancreas well. **(paper)**
- Pretraining helped on PBMC 12k, but models did not perform well even on datasets seen in pretraining. **(paper)** Exact numbers: not extracted.

## 6. Critical analysis

- Limitations (paper): only two models; zero-shot only.
- Weaknesses I see (my take): newer FMs (UCE, TranscriptFormer, scFoundation) not covered; results may shift with scale.
- What would change my mind: a newer FM beating HVG+scVI on the same protocol.

## 7. Relevance to my research

- Relation to [proposal](../00-plan/research-proposal.md): justifies always reporting PCA/linear baselines (Master's metrics) and fine-tuning over frozen embeddings.
- Hypotheses:
  - H-ZS1 (**E2.1**): rerun this protocol on UCE/TranscriptFormer: they still lose to HVG+scVI on AvgBio. Null: they win beyond CI.
  - H-ZS2 (**E2.2**): the FM-vs-baseline gap closes only at ≤1k labels after fine-tuning. Null: no crossover.

## 8. Related work

scGPT, Geneformer, scIB, PertEval-scFM, UCE.

## Resources

- [Paper](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-025-03574-x) · [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12007350/) · [bioRxiv 2023](https://www.biorxiv.org/content/10.1101/2023.10.16.561085)
