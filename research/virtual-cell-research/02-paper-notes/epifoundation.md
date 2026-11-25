# EpiFoundation: scATAC foundation model supervised by paired gene expression (peak-to-gene alignment)

> Wu J., Wan C., Ji Z., Zhou Y., Hou W. *EpiFoundation: A Foundation Model for Single-Cell ATAC-seq via Peak-to-Gene Alignment*. bioRxiv (2025, v2). DOI: [10.1101/2025.02.05.636688](https://doi.org/10.1101/2025.02.05.636688) · [PMC11839112](https://pmc.ncbi.nlm.nih.gov/articles/PMC11839112/) · Code: [UCSC-VLAA/EpiFoundation](https://github.com/UCSC-VLAA/EpiFoundation)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: read (PMC full text)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a transformer (6 layers, 512 dimensions) over a cell's *non-zero* peaks only. Instead of masked-peak reconstruction, it is pretrained to predict paired binarized gene expression from the [CLS] embedding. **(paper)**
- **Why it matters for the virtual cell**: it uses RNA as the supervision signal for ATAC, so the ATAC embedding is forced to encode *gene-regulatory* information. That is the property the Master's hypothesis assumes ATAC has. **(my take)**
- **Main claim**: state-of-the-art annotation and batch correction, and better expression prediction than Signac gene activity. **(paper)**

## Research summary

- **Context (already known)**: Masked reconstruction of sparse binary peaks mostly learns depth and cell-type structure, and paired RNA offers a denser target. **(paper)**
- **Gap (not known)**: Whether supervising an ATAC model with paired gene expression yields better scATAC embeddings and expression prediction. **(paper)**
- **Hypothesis / what they try to determine**: Predicting binarized gene expression from the ATAC [CLS] embedding (peak-to-gene alignment) forces the embedding to encode regulatory information. **(paper)**
- **Approach (how they investigated)**: A 6-layer transformer over non-zero peaks pretrained on MiniAtlas (>100k Multiome cells, 19 tissues, 56 cell types), evaluated on annotation, batch correction and expression prediction. **(paper)**
- **Results (what they found)**: Annotation accuracy 0.914 (kidney), 0.884 (PBMC), 0.842 (all tissues); best NMI on kidney and PBMC batch correction; PBMC expression prediction Spearman 0.42 vs 0.16 for Signac gene activity. **(paper)**
- **Implications (what it means)**: RNA supervision is a clean way to make an ATAC context encoder regulatory by design, but the only expression baseline is weak and the pretraining target matches the test task. **(my take)**

## 2. Problem it solves

Masked reconstruction on sparse binary peaks learns mostly depth and cell-type structure. Dense, paired RNA gives a richer target. **(paper)**

## 3. Method

- Input: binarized peaks, keeping only the non-zero ones (subsampled to 12k). Each token is a peak embedding plus a chromosome embedding. There is a [CLS] token, and a batch embedding is added only during pretraining. **(paper)**
- Objective: for each cell, sample 8,000 genes balanced between expressed and unexpressed, then predict binary expression from [CLS] plus a gene embedding with cross-entropy. **(paper)**
- Ablation (kidney NMI): 0.5681 for the full model, 0.4695 without the batch label, 0.4354 without the chromosome embedding. **(paper)**

## 4. Data

**MiniAtlas**: more than 100k 10x Multiome cells covering 19 tissues and 56 cell types (GEO + ENCODE). It was processed uniformly (Cell Ranger ARC, MACS2, Signac) with DISCO-based labels. **(paper)** This is a candidate entry for datasets.md.

## 5. Evaluation and results

- Annotation, accuracy / ROC-AUC: Kidney 0.914 / 0.987, PBMC 0.884 / 0.976, BMMC 0.762 / 0.976, all tissues 0.842 / 0.978. **(paper)**
- Batch correction: best NMI on kidney (0.568) and PBMC (0.738), but results are mixed across metrics. scANVI wins iso-label on kidney. **(paper)**
- Expression prediction (PBMC): MSE 6.76 vs 10.21 and Spearman 0.42 vs 0.16, against Signac gene activity. **(paper)**
- The only expression baseline is gene activity, which is weak. There is no comparison with BABEL, scButterfly, or a kNN-in-RNA baseline. **(my take)**

## 6. Critical analysis

- Limitations (paper): limited to two modalities. The future plan is to add sequence.
- Weaknesses I see (my take): the pretraining target is the same quantity as the downstream expression task, so the comparison against gene activity is not equal footing. Binarizing expression discards dose. With 100k cells the model is small-data for a "foundation" model.
- What would change my mind: if the expression-supervised embeddings beat masked-reconstruction embeddings (ChromFound, Atacformer) on E3.1's *regulatory* genes specifically, RNA supervision is the right choice for the context encoder.

## 7. Relevance to my research

- **Hypotheses**: this is the cleanest design for the ATAC context encoder in E4.1. Because of the peak-to-gene objective, the embedding is about regulation by construction. It is also the comparison point for E3.1.
- **Hypotheses worth testing**:
  - H-EF1: RNA-supervised ATAC embeddings give a larger gain than self-supervised ones (ChromFound/Atacformer) on E3.1 for high-tau genes. *Null*: equal or smaller.
  - H-EF2: RNA supervision leaks cell-type identity. A linear probe for cell type on EpiFoundation embeddings is at least as accurate as on RNA PCA, and in E4.1 it adds nothing beyond a cell-type one-hot. *Null*: E4.1 gain beyond the one-hot.

## 8. Related work

[ChromFound](chromfound.md), [Atacformer](atacformer.md), [EpiAgent](epiagent.md), [SCARF](scarf.md), CLM-Access, BABEL.

## Resources

- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11839112/
- Weights: https://huggingface.co/UCSC-VLAA/EpiFoundation
