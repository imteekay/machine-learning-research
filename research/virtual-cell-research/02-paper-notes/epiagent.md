# EpiAgent: a foundation model for single-cell epigenomics with in-silico cis-regulatory knockouts

> Chen X. et al. *EpiAgent: foundation model for single-cell epigenomics*. Nature Methods (2025). DOI: [10.1038/s41592-025-02822-z](https://doi.org/10.1038/s41592-025-02822-z) · Preprint: [bioRxiv 2024.12.19.629312](https://www.biorxiv.org/content/10.1101/2024.12.19.629312) · Code: [PyPI `epiagent`](https://pypi.org/project/epiagent/)
>
> PDF not committed (paywalled). The publisher page could not be fetched, so these notes come from the abstract and secondary summaries. Author list beyond the first author was not checked.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context), [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed (abstract-only; bioRxiv and publisher full text not retrievable on 2026-11-25)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a transformer that is pretrained on about 5M scATAC-seq cells (Human-scATAC-Corpus, about 35B tokens). Each cell becomes a compact "cell sentence" of accessible cis-regulatory elements (cCREs), and the model uses bidirectional attention. **(paper)**
- **Why it matters for the virtual cell**: it is one of the few chromatin foundation models that claims *perturbation* tasks: in-silico knockout of cCREs, and prediction of the cell state that follows. It is the most direct chromatin-side competitor to the Master's model. **(my take)**
- **Main claim**: a single pretrained model supports zero-shot annotation, integration, and in-silico cCRE knockout, as well as modeling of cell-state changes under perturbation. **(paper)**

## Research summary

- **Context (already known)**: scATAC-seq is high-dimensional and sparse, and analysis relied on per-dataset tools (ArchR, SnapATAC2, PeakVI) with no reusable pretrained representation. **(paper)**
- **Gap (not known)**: Whether a foundation model pretrained on millions of scATAC cells can support zero-shot tasks and in silico cis-regulatory perturbation. **(paper)**
- **Hypothesis / what they try to determine**: A transformer over per-cell sentences of accessible cCREs, pretrained at scale, supports annotation, integration, in silico cCRE knockout and cell-state prediction. **(paper)**
- **Approach (how they investigated)**: Pretrain on the Human-scATAC-Corpus (~5M cells, ~35B tokens), then evaluate zero-shot annotation, cCRE in silico KO and perturbation/state-change modelling. **(paper)** Only the abstract and secondary summaries were available; the full text was not fetchable.
- **Results (what they found)**: Reported to support all these tasks with one model; baselines and numbers were not retrieved. **(paper)**
- **Implications (what it means)**: A candidate chromatin context encoder, but token deletion is not a causal intervention, so its KO claim needs checking against CRISPRi enhancer screens. **(my take)**

## 2. Problem it solves

scATAC-seq matrices are huge (hundreds of thousands of peaks) and very sparse. Before EpiAgent, scATAC analysis relied on per-dataset tools (ArchR, SnapATAC2, PeakVI), with no reusable pretrained representation. **(paper)**

## 3. Method

- Cell sentences are built from accessible cCREs, and the tokens are cCREs, probably ranked by accessibility. Exact ranking details are not available in the abstract. **(paper / abstract-only)**
- A bidirectional transformer, reported by Atacformer to have roughly 1 to 1.5B parameters. That figure comes from a third party and was not verified. **(paper)**
- In-silico KO: remove cCRE tokens from the sentence and read out the change in the embedding or predicted state. **(paper)**

## 4. Data

- **Human-scATAC-Corpus**: about 5M cells and about 35B tokens. It also has its own database paper ([bioRxiv 2025.09.05.674505](https://www.biorxiv.org/content/10.1101/2025.09.05.674505)). **(paper)** This is a candidate entry for datasets.md.

## 5. Evaluation and results

Abstract-only. The tasks are zero-shot annotation, cCRE in-silico KO, and perturbation/state-change modeling. I could not retrieve the baselines or numbers.

## 6. Critical analysis

- Limitations (paper): not retrieved (abstract-only).
- Weaknesses I see (my take): deleting a cCRE token is not a causal intervention. It is a change to the input distribution, so the result may only show what the model associates with that region rather than what the region causes. The model is also very large for an ATAC model; Atacformer reports similar clustering with far fewer parameters.
- What would change my mind: if in-silico cCRE KO predictions agree with CRISPRi enhancer screens (for example, Gasperini et al. in K562) above a distance-to-TSS baseline, the KO task is meaningful.

## 7. Relevance to my research

- **Hypotheses**: EpiAgent is a ready-made chromatin context encoder for the Master's H1 (epigenetic context reduces OOD error), so it is a drop-in for the ATAC branch in E4.1. Its in-silico KO is directly testable against E3.3 (does baseline accessibility predict response?).
- **Hypotheses worth testing**:
  - H-EA1: frozen EpiAgent embeddings used as context in E4.1 reduce unseen-gene-KO error versus RNA-only context. *Null*: no gain beyond a cell-type one-hot.
  - H-EA2: in-silico cCRE KO effect sizes rank-correlate (Spearman > 0.3) with measured CRISPRi enhancer effects, and they beat a 1/distance-to-TSS baseline. *Null*: no better than the distance baseline.

## 8. Related work

[ChromFound](chromfound.md), [Atacformer](atacformer.md), [EpiFoundation](epifoundation.md), [SCARF](scarf.md), scBasset, PeakVI.

## Resources

- Nature Methods: https://www.nature.com/articles/s41592-025-02822-z
- Preprint: https://www.biorxiv.org/content/10.1101/2024.12.19.629312
- PyPI: https://pypi.org/project/epiagent/
