# scBERT: BERT-style pretrained transformer for single-cell cell type annotation

> Yang F, Wang W, Wang F, Fang Y, Tang D, Huang J, Lu H, Yao J. *scBERT as a large-scale pretrained deep language model for cell type annotation of single-cell RNA-seq data*. Nature Machine Intelligence 4, 852–866 (2022). DOI: [10.1038/s42256-022-00534-z](https://doi.org/10.1038/s42256-022-00534-z) · bioRxiv: [2021.12.05.471261](https://www.biorxiv.org/content/10.1101/2021.12.05.471261) · Code: [TencentAILabHealthcare/scBERT](https://github.com/TencentAILabHealthcare/scBERT)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a BERT-style transformer pretrained by masked expression prediction on unlabeled scRNA-seq (PanglaoDB), then fine-tuned for cell type annotation. **(paper)**
- **Why it matters for the virtual cell**: the first scFM; the simplest example of the "pretrain on cells, fine-tune on a task" recipe that scGPT and Geneformer extend. **(my take)**
- **Main claim**: scBERT gives strong cell type annotation, detects novel cell types, is robust to batch effects and is interpretable, using all genes without HVG selection or dimensionality reduction. **(paper)**

## 2. Problem it solves

Annotation methods depend on marker lists or HVG selection and transfer poorly across batches. Pretraining on large unlabeled data could learn gene-gene interactions useful for annotation. **(paper)**

## 3. Method

- Each gene is a token. Token embedding = **gene2vec** embedding (gene identity, standing in for position) + **binned expression** embedding. **(paper)** Default bins = 7 (`num_tokens`, suggested 5/7/9). **(code)**
- Encoder: **Performer** (linear-attention transformer), so that ~16k genes fit in one sequence without HVG selection. Defaults dim=200, depth=6, heads=10. **(code)**
- Pretraining: mask expression values and reconstruct. Fine-tuning: classifier head on cell embedding. **(paper)**
- Novel cell types: flag a cell as "unknown" if max predicted probability < threshold (default 0.5). **(code)**
- Input preprocessing: map symbols to an NCBI Gene list (Jan 2020), `normalize_total` + `log1p`. **(code)**

## 4. Data

- Pretraining: unlabeled scRNA-seq from PanglaoDB. **(paper)** Exact cell count not checked (abstract-only; often quoted as ~1M cells **(uncertain)**).
- Fine-tuning and benchmarks: multiple annotated datasets; specific list not checked.

## 5. Evaluation and results

Abstract-only for numbers: I could not access the full text (Nature paywall; bioRxiv rate-limited; README has no metrics). The abstract claims superior performance on annotation, novel cell type discovery, batch robustness and interpretability. **(paper)** Baselines and exact accuracy / F1 not checked.

## 6. Critical analysis

- Limitations (paper): not checked.
- Weaknesses I see (my take):
  - Annotation is a task where linear models and PCA + logistic regression are already strong; gains need to be shown against those.
  - Later independent re-evaluations (not read here) have reported that scBERT's advantage shrinks with class imbalance and smaller fine-tuning sets **(uncertain)**.
  - Binning to 7 levels discards a lot of quantitative information; this matters for perturbation prediction, where effects are often small shifts.
  - Max-softmax thresholding for novel types is a weak OOD detector.
- What would change my mind: a fair comparison on the same splits where scBERT beats logistic regression on PCA by more than the seed variance, especially on rare cell types.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: scBERT is listed as a foundational model in the Master's plan. It is a candidate backbone for the Phase 2 rung (do scFM embeddings beat PCA?), but its coarse binning makes it a poor fit for predicting subtle knockout effects in E4.1.
- Hypotheses worth testing:
  - **H-SCB-1 (E2.1/E2.2)**: fine-tuned scBERT beats PCA + logistic regression on annotation only when labeled cells per class are below ~100; with more labels the gap is within seed variance.
  - **H-SCB-2 (binning cost)**: re-encoding a perturbation dataset with 7 bins and then measuring how many known DE genes still differ between control and knockout cells. If many effects vanish after binning, binned-token models cannot represent them (relevant for backbone choice in E4.1).
  - **H-SCB-3 (OOD detection)**: on a held-out cell type, max-softmax AUROC for "novel" detection is lower than a simple kNN-distance detector on PCA.

## 8. Related work

- [Geneformer](geneformer.md) (rank tokens), [scGPT](scgpt.md) (binned values + generative pretraining), [scPoli](scpoli.md) (reference mapping without a transformer), [Transformers in single-cell omics](transformers-single-cell-review.md).

## Resources

- Code: https://github.com/TencentAILabHealthcare/scBERT
- bioRxiv: https://www.biorxiv.org/content/10.1101/2021.12.05.471261
