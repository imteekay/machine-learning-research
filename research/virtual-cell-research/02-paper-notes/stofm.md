# SToFM: a multi-scale (gene, micro, macro) foundation model for spatial transcriptomics

> Zhao, Luo, Yang, Zhong, Zhou & Nie. *SToFM: a Multi-scale Foundation Model for Spatial Transcriptomics*. ICML (2025). arXiv: 2507.11588 · Code: [PharMolix/SToFM](https://github.com/PharMolix/SToFM)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: read (arXiv HTML)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a Geneformer cell encoder (gene scale) feeding an SE(2)-invariant transformer over ~1,000-cell sub-slices (micro scale) augmented with ~50 cluster-level "virtual cells" (macro scale) **(paper)**; pretrained on the new SToCorpus-88M.
- **Why it matters for the virtual cell**: it separates *what a cell is* (encoder) from *where it is* (geometry-aware transformer). That split maps directly onto the PhD's generative vs spatial modules.
- **Main claim**: multi-scale modeling beats single-cell FMs and Nicheformer on segmentation, annotation, deconvolution and imputation, with F1 gains of 11.6% (segmentation) and 14.3% (annotation) over the best baseline **(paper)**.

## Research summary

- **Context (already known)**: Spatial FMs such as Nicheformer encode one cell at a time or ignore geometric invariance. **(paper)**
- **Gap (not known)**: Whether jointly modeling gene, microenvironment and tissue scales improves spatial transcriptomics tasks. **(paper)**
- **Hypothesis / what they try to determine**: That a multi-scale (gene, micro, macro) SE(2)-invariant model beats single-scale FMs. **(paper)**
- **Approach (how they investigated)**: Geneformer cell encoder, SE(2) transformer over ~1,000-cell sub-slices with Leiden-cluster virtual cells, pretrained on SToCorpus-88M. **(paper)**
- **Results (what they found)**: Wins on segmentation, annotation, deconvolution and imputation, e.g. embryo cross-slice macro-F1 0.459 vs 0.399 (CellPLM). **(paper)**
- **Implications (what it means)**: Multi-scale modeling helps over FM baselines, but without simple spatial-smoothing baselines the transformer's added value is unproven. **(my take)**

## 2. Problem it solves

ST needs gene-level, microenvironment and tissue-morphology information at once; existing FMs encode one cell at a time (Nicheformer) or ignore geometry invariance **(paper)**.

## 3. Method

- **Gene scale**: Geneformer domain-adapted on ST cells with masked gene modeling + contrastive learning **(paper)**.
- **Micro scale**: slices split into ~1,000-cell sub-slices by coordinates **(paper)**.
- **Macro scale**: Leiden clustering on PCA embedding + position (α=0.8); each cluster becomes a "virtual cell" token appended to every sub-slice **(paper)**.
- **SE(2) Transformer** (from Uni-Mol): pairwise distances through Gaussian basis as attention bias; invariant to 2D translation/rotation; 4 layers, hidden 256, 8 heads **(paper)**.
- **Objectives**: masked cell modeling (mask 10% of cell embeddings, MSE) and pairwise distance recovery (noise 10% of coordinates, reconstruct distances) **(paper)**.

## 4. Data

**SToCorpus-88M**: 1,912 slices, ~88M cells, six technologies (MERFISH, Xenium, CosMx, Stereo-seq, Slide-seqV2, SeekSpace), human + mouse, incl. unpublished SeekSpace data; cells <100 genes removed; ~1.5x Nicheformer's corpus; release planned **(paper)**. New dataset for [datasets](../03-data/datasets.md).

## 5. Evaluation and results

Head-tuning vs Geneformer, scGPT, Nicheformer, CellPLM **(paper)**:

| Task | SToFM | Best baseline |
|---|---|---|
| Embryo segmentation cross-slice macro-F1 | 0.459 | CellPLM 0.399 |
| DLPFC layer macro-F1 | 0.644 | CellPLM 0.595 |
| Mouse brain annotation Brain1 macro-F1 | 0.495 | CellPLM 0.419 |
| Liver deconvolution cross-slice Pearson | 0.755 | CellPLM 0.725 |
| Skin imputation cross-slice Pearson | 0.488 | CellPLM 0.399 |

Ablation: removing domain adaptation drops embryo cross-slice F1 0.459 -> 0.375; removing virtual cells -> 0.429 **(paper)**. Baselines are all FMs; no PCA/linear or spatial-smoothing baseline **(my take)**.

## 6. Critical analysis

- Limitations (paper): only three scales; no causal/regulatory modeling, ligand-receptor priors or histology.
- Weaknesses I see (my take): no simple baselines (PCA + neighbor-averaged expression, STAGATE/GraphST on segmentation). Macro "virtual cells" from Leiden clustering are a hand-built prior; gains may come from cluster smoothing, not learning. Cross-slice numbers are modest in absolute terms.
- What would change my mind: SToFM losing to "PCA of neighbor-averaged expression + linear head" on segmentation would say the transformer adds little.

## 7. Relevance to my research

- **Hypotheses**: PhD H requires a spatial module; SE(2) invariance with distance-biased attention is a principled default. But perturbation responses at a border are *directional* (proposal: "directional kernels"), which SE(2)-invariant distance-only attention cannot express without extra features **(my take)**.
- Hypotheses worth testing:
  - **H-a (E5.1)**: niche information comes mostly from neighbor averaging; an SToFM embedding beats "own embedding + mean of k neighbors" by >X macro-F1 on held-out slices, else the transformer is redundant.
  - **H-b (E5.2)**: distance-only attention under-predicts asymmetric bystander effects at tumor/immune borders vs a model with relative-direction features (Perturb-map).
  - **H-c (E5.3)**: macro virtual-cell tokens help most on cross-slice splits; ablation effect larger cross-slice than within-slice.

## 8. Related work

[Nicheformer](nicheformer.md), [scGPT-spatial](scgpt-spatial.md), [HEIST](heist.md), [SpatialFormer](spatialformer.md), [TERRA](terra.md), CellPLM.

## Resources

- arXiv: https://arxiv.org/abs/2507.11588
- Code: https://github.com/PharMolix/SToFM
