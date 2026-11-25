# RegFormer: GRN-structured Mamba foundation model

> Hu L, Zhang Y, et al. *RegFormer: a single-cell foundation model powered by gene regulatory hierarchies*. Nature Communications (2026). DOI: [10.1038/s41467-026-72198-x](https://doi.org/10.1038/s41467-026-72198-x) · Preprint: [bioRxiv 2025.01.24.634217](https://www.biorxiv.org/content/10.1101/2025.01.24.634217)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Only the abstract or secondary sources were read for this note (**abstract-only**); details beyond those are marked as missing rather than guessed.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: an scRNA FM that orders/structures genes by GRN regulatory hierarchy and uses Mamba blocks; ~50M parameters, pretrained on 22M human cells, with separate embeddings for expression values and gene identity. **(preprint, via secondary summary)**
- **Why it matters for the virtual cell**: an explicit regulatory inductive bias, the same bet as my H1 (but via prior GRNs rather than measured ATAC). **(my take)**
- **Main claim**: beats scGPT and Geneformer on annotation, GRN reconstruction, perturbation and drug response. **(preprint)**

## Research summary

- **Context (already known)**: Generic scFMs (scGPT, Geneformer) treat genes as a flat or expression-ranked sequence, and regulatory relationships are known from GRN resources. **(paper)**
- **Gap (not known)**: Whether injecting GRN hierarchy into a single-cell FM improves representation and downstream tasks; the exact gains and the effect of the prior are not available in the abstract. **(my take)**
- **Hypothesis / what they try to determine**: Ordering and structuring genes by regulatory hierarchy, with a Mamba backbone, yields a better FM than hierarchy-free models. **(paper)**
- **Approach (how they investigated)**: Pretrain a ~50M-parameter Mamba model on 22M human cells with GRN-derived gene ordering, then compare with scGPT and Geneformer on annotation, GRN reconstruction, perturbation and drug response. **(paper, abstract-level)**
- **Results (what they found)**: Reported wins over scGPT and Geneformer on four task families; numbers not available in abstract. **(paper)**
- **Implications (what it means)**: If the gain survives a shuffled-GRN control it supports regulatory inductive bias; without that ablation it may just reflect Mamba. Full text unavailable (bioRxiv returned HTTP 429). **(my take)**

## 2. Problem it solves

Generic FMs treat genes as an unordered set and may not learn regulatory logic. **(preprint)**

## 3. Method

GRN priors define a hierarchy used to order the gene sequence for a Mamba state-space model. **(preprint, abstract-level)** Exact construction: abstract-only.

## 4. Data

22M human cells. **(preprint)**

## 5. Evaluation and results

Wins over scGPT/Geneformer on four task families. **(preprint)** Numbers, mean baselines: abstract-only.

## 6. Critical analysis

- Limitations (paper): not extracted.
- Weaknesses I see (my take): GRN priors are cell-type-agnostic; a random-hierarchy ablation is the key control.
- What would change my mind: a shuffled-GRN ablation performing as well (would mean the gain is Mamba, not biology).

## 7. Relevance to my research

- Relation to [proposal](../00-plan/research-proposal.md): direct competitor to H1's mechanism; informs **E4.1**/**E4.2** design.
- Hypotheses:
  - H-RF1: on unseen knockouts, RegFormer with the true GRN beats the same model with a degree-preserving shuffled GRN. Null: no difference.
  - H-RF2 (**E4.1**): cell-type-specific ATAC-derived GRNs outperform static priors as the RegFormer hierarchy. Null: no gain.

## 8. Related work

scPRINT, GRNFormer, scGPT, Geneformer, SCENIC+.

## Resources

- [Paper](https://www.nature.com/articles/s41467-026-72198-x) · [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.01.24.634217)
