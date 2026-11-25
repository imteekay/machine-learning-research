# SCimilarity: metric-learning foundation model for cell search across atlases

> Heimberg G, Kuo T, et al. *A cell atlas foundation model for scalable search of similar human cells*. Nature 638, 1085–1094 (2025). DOI: [10.1038/s41586-024-08411-y](https://doi.org/10.1038/s41586-024-08411-y) · Code: [Genentech/scimilarity](https://github.com/Genentech/scimilarity)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Only the abstract or secondary sources were read for this note (**abstract-only**); details beyond those are marked as missing rather than guessed.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a metric-learning model giving one representation to query tens of millions of cells for transcriptionally similar ones. **(paper)**
- **Why it matters for the virtual cell**: search links patient cell states to in vitro systems where they can be perturbed, a practical bridge between Phase 4 experiments and Phase 6 patients. **(my take)**
- **Main claim**: querying a 23.4M-cell, 412-study atlas with ILD fibrosis-associated macrophages/fibroblasts finds similar states in other fibrotic diseases; the top in vitro hit (a 3D hydrogel) was experimentally confirmed to reproduce the macrophage state. **(paper)**

## 2. Problem it solves

Atlases are too large and heterogeneous to search for "cells like this one" across studies and tissues. **(paper)**

## 3. Method

Metric learning (triplet-style with annotated cell types, plus reconstruction, per my recollection) producing an embedding and kNN index for fast search. **(paper: metric learning; details my take, abstract-level)**

## 4. Data

23.4M cells from 412 scRNA-seq studies (the HF model card says training used 22.7M cells from 399 studies, so training and query atlases may differ). **(secondary sources)** Weights, embeddings, kNN graphs on Zenodo.

## 5. Evaluation and results

Case study: fibrosis-associated macrophages across diseases; experimental validation of the hydrogel in vitro hit. **(paper)** Benchmark numbers vs scVI/scGPT: abstract-only.

## 6. Critical analysis

- Limitations (paper): not extracted.
- Weaknesses I see (my take): supervised metric learning on curated labels bakes annotation choices into similarity; novel states may be pulled toward known types.
- What would change my mind: search recall on held-out disease states with no label in training.

## 7. Relevance to my research

- Relation to [proposal](../00-plan/research-proposal.md): enables choosing perturbation contexts that resemble patient cells (Phase 4→6), and is a donor-identifiability concern for **E1.3**.
- Hypotheses:
  - H-SC1: perturbation responses measured in the in vitro system ranked most similar by SCimilarity transfer to patient-like cells better (lower E-distance in **E4.1**-style evaluation) than in a random/lower-ranked cell line. Null: no rank-dependence.
  - H-SC2 (**E1.3**): SCimilarity nearest neighbours re-identify the donor above chance across studies. Null: chance-level.

## 8. Related work

UCE, scVI/scArches, CELLxGENE Census.

## Resources

- [Paper](https://www.nature.com/articles/s41586-024-08411-y) · [PyPI](https://pypi.org/project/scimilarity/) · [HF model](https://huggingface.co/deeplife/scimilarity_model) · [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2023.07.18.549537)
