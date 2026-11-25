# CellFM: an 800M-parameter foundation model on 100M human cells

> Zeng Y, Xie J, Shangguan N, Wei Z, Li W, Su Y, Yang S, Zhang C, Zhang J, Fang N, Zhang H, Lu Y, Zhao H, Fan J, Yu W, Yang Y. *CellFM: a large-scale foundation model pre-trained on transcriptomics of 100 million human cells*. Nature Communications (2025). DOI: [10.1038/s41467-025-59926-5](https://doi.org/10.1038/s41467-025-59926-5) · PMC12092794 · Code: [biomed-AI/CellFM](https://github.com/biomed-AI/CellFM)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Only the abstract or secondary sources were read for this note (**abstract-only**); details beyond those are marked as missing rather than guessed.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: an 800M-parameter human-only scFM trained on a curated set of 100M human cells, using a modified RetNet backbone on MindSpore. **(paper)**
- **Why it matters for the virtual cell**: a test of the "scale" thesis on human-only data; useful to argue whether size or curation drives gains. **(my take)**
- **Main claim**: outperforms existing models in cell annotation, perturbation prediction, gene function prediction and gene–gene relationship capture. **(paper)**

## 2. Problem it solves

Human scFMs were limited in training data and parameter count; noise, batch effects and sparsity call for a unified representation. **(paper)**

## 3. Method

Modified RetNet (retention instead of attention, linear-cost) for efficiency; 800M parameters; trained on MindSpore (Huawei Ascend). **(paper)** Objective details: abstract-only.

## 4. Data

~100M human cells collected from public sources. **(paper)** Exact sources/QC: abstract-only.

## 5. Evaluation and results

Annotation, perturbation prediction, gene function prediction, gene–gene relations; claims to beat existing models. **(paper)** Numbers and baseline strength: abstract-only.

## 6. Critical analysis

- Limitations (paper): not extracted.
- Weaknesses I see (my take): MindSpore dependency hinders reuse; perturbation gains rarely survive mean/linear baselines in independent benchmarks.
- What would change my mind: CellFM beating the training-mean baseline on unseen single-gene knockouts (Systema-corrected).

## 7. Relevance to my research

- Relation to [proposal](../00-plan/research-proposal.md): informs **E1.2** (curation vs size) and the RNA-only arm of **E4.1**.
- Hypotheses:
  - H-CF1 (**E1.2**/**E2.2**): at matched compute, a model trained on a curated 10M subset matches CellFM on annotation macro-F1. Null: CellFM wins beyond CI.
  - H-CF2 (**E4.4**): CellFM's perturbation advantage vanishes after Systema correction for systematic variation. Null: advantage persists.

## 8. Related work

scFoundation, Geneformer V2, Tahoe-x1, scGPT.

## Resources

- [Paper](https://www.nature.com/articles/s41467-025-59926-5) · [Code](https://github.com/biomed-AI/CellFM)
