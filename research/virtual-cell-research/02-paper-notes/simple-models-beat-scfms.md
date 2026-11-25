# Simple models vs scFMs: normalization + linear pipelines match or beat foundation models

> Souza H, Mehta P. *Simple models outperform single-cell foundation models on downstream benchmarks* (listed in the roadmap as "Parameter-free representations outperform single-cell foundation models on downstream benchmarks"). arXiv preprint (2026). arXiv: [2602.16696](https://arxiv.org/abs/2602.16696) · Code: none linked on arXiv

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: read
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: rank-based normalization (scTOP) plus ANOVA gene selection, whitened PCA and logistic regression, compared zero-shot against TranscriptFormer, ESM2-CE, UCE, scVI and Geneformer on the same splits and classifier. **(paper)**
- **Why it matters for the virtual cell**: it sets the baseline bar. Any claim that a pretrained representation "knows biology" has to beat a few-line linear pipeline that needs no pretraining. **(my take)**
- **Main claim**: simple pipelines reach SOTA or near-SOTA on cell-type, disease-state and cross-species tasks, and win on cross-species OOD. **(paper)**

## 2. Problem it solves

scFM papers usually compare against weak baselines such as raw HVG + PCA. This paper asks how far careful normalization and linear methods go on the benchmarks the scFMs report. **(paper)**

## 3. Method

- scTOP: per-cell total-count normalization, log2(x+1), within-cell gene ranks converted to percentiles and then to z-scores with the inverse normal CDF. One reference vector per cell type, and a de-correlated projection (inverse overlap matrix) to score query cells. **(paper)**
- Extended pipeline: scTOP preprocessing, then ANOVA F-test gene selection on training data, per-gene standardization, PCA with 220 whitened components and class-balanced L2 logistic regression. For COVID-19 it adds Leiden clusters with one classifier per cluster. **(paper)**
- TF→gene associations: binarized expression and a PMI-like enrichment in TF-active cells, with BH correction. **(paper)**

## 4. Data

- Spermatogenesis across 8 mammals (human to platypus), 7 cell types.
- Tabula Sapiens 2.0 (26 tissues).
- COVID-19 lung/immune data (4 donors, infected vs uninfected).
- STRING v12.5 as TF–gene validation. **(paper)** No new dataset.

## 5. Evaluation and results

- Cross-species: mean macro F1 is **0.679** (scTOP) vs 0.639 (Geneformer, best FM) and 0.515 (TF-Sapiens). It is the best representation for 6 of 8 source species. **(paper)**
- Tabula Sapiens 2.0: mean macro F1 is **0.887** vs **0.911** for TF-Sapiens. TF-Sapiens is higher in 294 of 330 tissue × cell-type combinations, but the mean gap is only −0.027. **(paper)**
- COVID-19: macro F1 0.850, above most FMs and competitive with TF-Sapiens. **(paper)**
- TF–gene: 63% STRING support, the same as the FM-derived associations, and better in the top-k (75/70/56% vs 62/58/50%). **(paper)**
- Baselines are strong by construction: this paper *is* the strong baseline. **(my take)**

## 6. Critical analysis

- Limitations (paper): limited to current atlases and benchmarks. Dynamics and multimodal tasks may need more expressive models. scTOP alone is poor on TS2.0, so the ANOVA/PCA/classifier steps matter. Hyperparameters were grid-searched on 3 splits, so the FDR is approximate. The TF analysis covers only 5 TFs in bone marrow.
- Weaknesses I see (my take): the in-distribution TS2.0 result is a small *loss* for the simple pipeline, so "outperform" in the title is task-dependent. ANOVA gene selection uses labels and FM embeddings are used zero-shot, which is a mild asymmetry in supervision. All tasks are classification. Nothing tests perturbation prediction, which is where the virtual-cell claim lives.
- What would change my mind: an scFM that beats this pipeline by a CI-separated margin on *held-out tissues or species*, not only in-distribution.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: it supports the "context, not scale" framing. If identity is mostly linear in well-normalized expression, the gain for the virtual cell must come from extra context (ATAC, niche), not from bigger expression-only pretraining. **(my take)**
- Hypotheses worth testing:
  - **E2.1 upgrade**: replace the HVG + PCA baseline with scTOP + whitened PCA + LR. H: the scGPT/Geneformer/UCE advantage in E2.1 shrinks to within CI. Null: the FM still wins by a CI-separated margin.
  - **E2.2**: H: the fine-tuning advantage only appears at ≤ 1k labels *if* the baseline is scTOP rather than raw PCA. Null: it disappears at all label sizes.
  - **E4.x baseline**: use rank-normalized z-score expression as the input to the linear perturbation baseline. H: it narrows the gap to deep perturbation models more than log1p input does.

## 8. Related work

Kedzierska et al. 2025 (zero-shot scFM limits), Ahlmann-Eltze 2025 (linear perturbation baselines), [Metric mirages / drifting islands](drifting-islands-embedding-metrics.md), [scaling evidence](scfm-pretraining-data-scaling.md), TranscriptFormer.

## Resources

- arXiv: https://arxiv.org/abs/2602.16696 (HTML v2)
