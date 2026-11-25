# Novae: graph-based foundation model for spatial transcriptomics

> Blampey Q, Benkirane H, Bercovici N, et al. *Novae: a graph-based foundation model for spatial transcriptomics data*. Nature Methods (2025) · DOI: 10.1038/s41592-025-02899-6 · Preprint: bioRxiv 10.1101/2024.09.09.612009 · Code: Python package `novae` (PyPI), weights on Hugging Face

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. This note is **abstract-only**: Nature login-walled and bioRxiv rate-limited; details from search snippets.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: self-supervised graph attention network over cell neighborhoods that yields niche (spatial domain) representations, usable across gene panels and technologies. **(paper, summary)**
- **Why it matters for the virtual cell**: a ready encoder for the "niche context" input of the PhD tissue-level model. **(my take)**
- **Main claim**: trained on ~29M cells, 18 tissues, three technologies; zero-shot spatial domain identification, built-in batch correction, downstream spatially variable genes and domain trajectories. **(paper, summary)**

## Research summary

- **Context (already known)**: Spatial transcriptomics domain-identification methods are mostly trained per dataset and gene panel, and single-cell FMs ignore tissue structure. **(paper)**
- **Gap (not known)**: Whether one self-supervised graph model can give zero-shot niche representations across panels, technologies and tissues is not established; full text not read. **(my take)**
- **Hypothesis / what they try to determine**: Whether a graph attention network pretrained on ~29M cells from 18 tissues and 3 technologies transfers zero-shot to spatial domain identification. **(paper)**
- **Approach (how they investigated)**: Self-supervised graph attention over cell neighborhoods, benchmarked on domain identification and downstream spatially variable genes and domain trajectories; benchmark details not retrieved (abstract-only). **(paper)**
- **Results (what they found)**: Reported zero-shot domain identification with built-in batch correction; no numbers or baselines seen. **(paper)**
- **Implications (what it means)**: If it holds, Novae is a ready niche encoder for tissue-level models, but whether niche embeddings add information beyond cell type is untested here. **(my take)**

## 2. Problem it solves

Spatial domain methods are tied to one dataset/panel and need retraining. **(paper)**

## 3. Method

Graph attention over cell graphs, multiple gene panels handled in one model; self-supervised objective (details not retrieved).

## 4. Data

29M cells, 18 tissues, 3 technologies (e.g. MERFISH/Xenium-style imaging platforms; **my take**, exact list not seen).

## 5. Evaluation and results

Abstract-only; domain-identification benchmarks and baselines not retrieved.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): domain labels are subjective; domain ID does not test whether niche embeddings predict perturbation effects.
- What would change my mind: Novae embeddings failing to add information over cell type in E5.1.

## 7. Relevance to my research

- Proposal link: PhD hypothesis (spatial context improves response prediction at borders) needs a niche encoder; Novae is a candidate frozen one.
- Hypotheses worth testing:
  - E5.1: Novae niche embeddings add predictive information for gene expression beyond cell type (falsified if neighbor-composition baseline matches).
  - E5.3: zero-shot Novae embeddings transfer to an unseen tissue with <10% loss in niche-prediction accuracy.

## 8. Related work

HEIST ([heist](heist.md)), CellCharter, Nicheformer.

## Resources

- [Nature Methods](https://www.nature.com/articles/s41592-025-02899-6) · [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.09.09.612009)
