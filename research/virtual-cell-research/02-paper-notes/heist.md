# HEIST: hierarchical graph foundation model (genes in cells in tissue) for spatial transcriptomics and proteomics

> Madhu, Rocha, Huang, Viswanath, Krishnaswamy & Ying. *HEIST: A graph foundation model for spatial transcriptomics and proteomics data*. arXiv (2025). arXiv: 2506.11152 · Code: [Graph-and-Geometric-Learning/HEIST](https://github.com/Graph-and-Geometric-Learning/HEIST) · Weights: [HF](https://huggingface.co/HirenMadhu/HEIST)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: read (arXiv HTML v4, first ~85%)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: two-level graph transformer — a Voronoi spatial cell graph, where each cell carries a gene co-expression graph (per cell type, by mutual information) — with intra- and cross-level message passing **(paper)**.
- **Why it matters for the virtual cell**: it encodes the hierarchy the PhD model needs (gene regulation inside cells, signaling between cells) and has no fixed gene vocabulary, so it can take proteomics panels.
- **Main claim**: SOTA on clinical outcome prediction, annotation and imputation; generalizes to spatial proteomics; ~8x faster than scGPT-spatial **(paper)**.

## 2. Problem it solves

Spatial FMs ignore intra-cell gene structure and depend on fixed vocabularies, so new panels/modalities need retraining **(paper)**.

## 3. Method

- Gene embeddings from rank-based sinusoidal encodings + graph updates (vocabulary-free) **(paper)**.
- Each layer: cell-level dense attention + GIN, gene-level sparse attention, then directional cross-level message passing **(paper)**.
- Objectives: contrastive over same-type spatial neighbors, cross-level cell-gene alignment, masked autoencoding of expression and coordinates, orthogonality regularizer, learnable loss weighting **(paper)**.

## 4. Data

22.3M cells, 124 slices, 15 organs: 13.3M 10x (Xenium), 8.7M Vizgen (MERFISH), 360K SEA-AD; single-cell-resolution only **(paper)**.

## 5. Evaluation and results

- Imputation (fine-tuned Pearson): placenta 0.821 vs CellPLM 0.801; skin 0.807 vs scGPT-spatial 0.740; zero-shot weak (0.350 skin) **(paper)**.
- Annotation F1: SEA-AD 0.995; trails scGPT-spatial on MERFISH lung (in scGPT-spatial's pretraining) **(paper)**.
- Clinical outcome AUC: best in 6/7 settings vs STAGATE, GraphST, scFoundation, Novae, CellPLM, scGPT-spatial, SPACE-GM **(paper)**.
- L-R prediction linear probe AUC 0.995 **(paper)**.
- Baselines include spatial non-FM methods — good — but no PCA/linear baseline on imputation **(my take)**.

## 6. Critical analysis

- Limitations (paper): mismatched pretraining data across baselines; abstract says five organs evaluated.
- Weaknesses I see (my take): co-expression graphs per cell type are correlational, not regulatory; calling them GRNs would be wrong. Clinical outcome tasks have few patients, so AUC CIs are likely wide. L-R AUC 0.995 suggests an easy negative set.
- What would change my mind: replacing gene graphs with random graphs of same degree barely changing results would say the hierarchy is decorative.

## 7. Relevance to my research

- **Hypotheses**: the Master's H1 says regulatory context (ATAC) grounds predictions; HEIST's gene-level graph is a slot where ATAC-derived regulatory graphs could replace MI co-expression **(my take)**. PhD H: cross-level message passing is a candidate spatial module.
- Hypotheses worth testing:
  - **H-a (E3.x/E5.1)**: swapping MI co-expression graphs for ATAC-informed TF->target graphs improves held-out imputation/annotation; falsified if equal to random graphs.
  - **H-b (E5.2)**: hierarchical model predicts neighbor responses to KO better than a flat cell graph (GNN on cell embeddings only).
  - **H-c**: vocabulary-free genes enable zero-shot transfer to an unseen panel at above-PCA performance.

## 8. Related work

[scGPT-spatial](scgpt-spatial.md), [SToFM](stofm.md), [NicheCompass](nichecompass.md), CellPLM, Novae, SPACE-GM.

## Resources

- arXiv: https://arxiv.org/abs/2506.11152
- Code: https://github.com/Graph-and-Geometric-Learning/HEIST
