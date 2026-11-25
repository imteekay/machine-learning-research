# scFoundation: a 100M-parameter read-depth-aware scRNA foundation model

> Hao M, Gong J, Zeng X, Liu C, Guo Y, Cheng X, Wang T, Ma J, Zhang X, Song L. *Large-scale foundation model on single-cell transcriptomics*. Nature Methods (2024). DOI: [10.1038/s41592-024-02305-7](https://doi.org/10.1038/s41592-024-02305-7) · Code: [biomap-research/scFoundation](https://github.com/biomap-research/scFoundation)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Only the abstract or secondary sources were read for this note (**abstract-only**); details beyond those are marked as missing rather than guessed.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a ~100M-parameter asymmetric transformer ("xTrimoscFoundation-α") over ~20,000 genes, pretrained on >50M human single-cell profiles. **(paper)**
- **Why it matters for the virtual cell**: it treats sequencing depth as an explicit input, which is the right inductive bias for a model that must generalize across protocols and clinical samples with very different depths. **(my take)**
- **Main claim**: state of the art on expression enhancement, tissue and single-cell drug response, perturbation prediction, annotation and gene-module inference. **(paper)**

## Research summary

- **Context (already known)**: Earlier scFMs truncate to HVGs or rank-encode genes, discarding absolute expression and depth. **(paper, abstract-level)**
- **Gap (not known)**: Whether a model over ~20k genes with explicit read-depth handling, pretrained at >50M cells, transfers across many tasks. **(paper)**
- **Hypothesis / what they try to determine**: An asymmetric encoder-decoder with depth-aware pretraining gives better representations for enhancement, drug response, perturbation and annotation. **(paper)**
- **Approach (how they investigated)**: Pretrain on >50M human profiles and evaluate on six task families, including GEARS-style perturbation. Details abstract-only. **(paper)**
- **Results (what they found)**: Claims state of the art across tasks; numbers not available in abstract. **(paper)**
- **Implications (what it means)**: Depth conditioning is a good bias, but gains may come from downstream heads; unverified. **(my take)**

## 2. Problem it solves

Earlier scFMs truncate to a few thousand HVGs or rank-encode genes, discarding absolute expression and depth. scFoundation tries to model the full gene set and the count scale at once. **(paper, abstract-level)**

## 3. Method

- Asymmetric "transformer-like" encoder–decoder: the abstract highlights the asymmetry and pretraining task design. **(paper)**
- From what I recall: the encoder sees only nonzero genes and the decoder sees all genes; a read-depth-aware masked objective conditions on source and target total counts so the model can "enhance" depth. **(my take, not verified; abstract-only)**

## 4. Data

>50M human scRNA-seq profiles covering ~20k genes. **(paper)** Sources and QC: abstract-only, not checked.

## 5. Evaluation and results

Tasks listed: gene expression enhancement, tissue drug response, single-cell drug response classification, single-cell perturbation prediction (plugged into GEARS-style pipelines), cell type annotation, gene module inference. **(paper)** Numbers and baselines: abstract-only, not extracted. Whether mean/linear baselines were used for perturbation is unknown. **(my take)**

## 6. Critical analysis

- Limitations (paper): not extracted (abstract-only).
- Weaknesses I see (my take): "SOTA on many tasks" claims for scFMs have repeatedly shrunk under independent benchmarks (see [zero-shot limitations note](zero-shot-scfm-limitations.md)); perturbation gains via downstream heads may come from the head, not the embedding.
- What would change my mind: depth-enhanced profiles improving a downstream task (e.g. DE recovery) on shallow clinical data beyond simple depth normalization + imputation baselines.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: the Master's H1 needs an RNA backbone; scFoundation is a candidate frozen/fine-tuned RNA encoder for the RNA-only arm of **E4.1**. Its depth conditioning matters for E4.4 (shortcut checks): depth is a known confounder of knockdown efficiency.
- Hypotheses worth testing:
  - H-sF1: in **E2.1**, scFoundation embeddings beat HVG+PCA on cell-type macro-F1 only on low-depth cells (bottom depth quartile). Null: no depth-dependent gap.
  - H-sF2: "read-depth enhancement" of shallow cells improves E-distance of perturbation predictions in **E4.1** vs. using raw counts. Null: no change beyond CI.

## 8. Related work

scGPT, Geneformer, GEARS, CellFM, [zero-shot limitations](zero-shot-scfm-limitations.md).

## Resources

- [Paper](https://www.nature.com/articles/s41592-024-02305-7) · [Code](https://github.com/biomap-research/scFoundation)
