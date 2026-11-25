# scGen: predicting single-cell perturbation responses with VAE latent arithmetic

> Lotfollahi M, Wolf FA, Theis FJ. *scGen predicts single-cell perturbation responses*. Nature Methods 16, 715–721 (2019). DOI: [10.1038/s41592-019-0494-8](https://doi.org/10.1038/s41592-019-0494-8) · Code: [theislab/scgen](https://github.com/theislab/scgen)
>
> PDF not committed (paywalled; preprint bioRxiv 478370).

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Claims marked **(uncertain)** are from memory of the paper and were not re-checked against the PDF.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read
- **Read on**: 2026-11-25
- **Experiment**: [E4.5](../04-experiments/E4.5-scgen-unseen-cell-type.md) · code [`scgen-mini/`](../scgen-mini/)

## 1. TL;DR

- **What**: train a VAE on single-cell expression; compute δ = mean z(stimulated) − mean z(control) on cell types where both are measured; add δ to the latent codes of a cell type seen only in control, and decode to predict its stimulated state. **(paper)**
- **Why it matters for the virtual cell**: the first widely used "unseen cell context" perturbation model, and the ancestor of CPA, scPreGAN, CellOT, etc. It defines the split (leave-one-cell-type-out) that E4.1 inherits. **(my take)**
- **Main claim**: latent arithmetic in a VAE beats arithmetic in gene space or PCA space and beats CVAE / style-transfer GAN baselines, across cell types, studies and species. **(paper)**

## 2. Problem it solves

Measuring every perturbation in every cell type is impossible. If a perturbation's effect is partly shared across types, a model could learn it on some types and transfer it to others, from control cells alone. **(paper)**

## 3. Method

- VAE: encoder and decoder MLPs (2 × 800 hidden, latent 100, BatchNorm, LeakyReLU, dropout 0.2), Gaussian latent, MSE reconstruction on log-normalised data, KL weight α very small. **(paper/code; exact α (uncertain))**
- δ from the training data after *balancing* cell types by resampling, so large types do not dominate. **(paper)**
- Prediction: decode z(ctrl cells of held-out type) + δ. **(paper)**
- Argument for nonlinearity: the VAE latent space is "more linear" for the perturbation than gene space, so a single vector transfers better. **(paper)** Why that would be true is not shown mechanistically. **(my take)**

## 4. Data

- Kang 2018 PBMC IFN-β (8 lupus donors, ~7 types; their main benchmark), Haber 2017 gut epithelium (Salmonella / H. poly infection), cross-species LPS (Hagai 2018), cross-study transfer. **(paper)**
- E4.5 uses only Kang 2018 from GEO GSE96583 batch 2 ([datasets](../03-data/)).

## 5. Evaluation and results

- Metric: R² of mean expression (predicted vs real stimulated cells) over all genes and over the top-100 DE genes; also distributions of single marker genes (e.g. ISG15). **(paper)**
- Headline: high R² for held-out CD4 T cells, lower for monocytes, VAE > linear baselines. **(paper; exact numbers (uncertain))**
- Baselines are linear (gene-space and PCA arithmetic) and generative (CVAE, GAN). No "no change" baseline in the headline plots, and no correction for the shared response. **(my take; (uncertain) whether a supplement has it)**

## 6. Critical analysis

- **Limitations (paper)**: needs the perturbation measured in some cell types; assumes the response is roughly shared; a single δ cannot represent responder subpopulations.
- **Weaknesses I see (my take)**:
  - R² on *means* is flattered by baseline expression; "no change" already scores 0.85–0.91 on all genes for held-out lymphocytes in E4.5. Even top-100 DE R² is dominated by the ISGs every type induces.
  - In E4.5 the shared response explains 69–84% of each type's Δ variance (leave-one-type-out). What would really test transfer is the type-specific part.
  - A VAE decoder smooths cells, so decoded cells have less variance than real ones. Distributional metrics (energy distance) can penalise it even when the mean is right.
- **What would change my mind**: a residual (Systema-style, projected) correlation clearly above a shuffled-δ control across several held-out types.

## 7. Relevance to my research

- Directly the "unseen cell context" arm of Master's H1 (E4.1): scGen-style transfer is a baseline that any ATAC-conditioned model must beat.
- Hypotheses: (a) VAE skill over gene-space shift is mostly *amplitude* (how strongly a type responds), not *which genes*, tested in E4.5; (b) conditioning δ on cell-type features (e.g. baseline IFNAR/STAT1 levels, chromatin accessibility at ISG promoters) could predict amplitude and specific genes (E4.1/E4.2 territory).

## 8. Related work

CPA (Lotfollahi 2023, compositional, dose/combination), CellOT (Bunne 2023, optimal transport), scPreGAN, Systema (systematic-variation correction), Ahlmann-Eltze 2025 (linear baselines match deep models for perturbation prediction).

## Resources

- Code: https://github.com/theislab/scgen (scvi-tools based)
- Data: GEO GSE96583; pertpy `pt.dt.kang_2018()`
