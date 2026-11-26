# Learnings ledger

Biology and ML lessons from every experiment, grouped by domain. Each experiment README has its own "Learnings ledger" section; copy its bullets here and tag each one *new*, *confirms Ex.y* or *contradicts Ex.y*.

## Biology

### Assay physics and count statistics
- Droplet (UMI) zeros are explained by depth plus NB overdispersion, with no excess dropout. ZINB gains < 0.13% of the LL. (E0.1, *new*)
- Most overdispersion of HVGs is cell-state heterogeneity: marginal θ ≈ 0.1, but θ ≈ 1 given the latent. (E0.1, *new*)
- Deeper chemistry (v3) makes overdispersion visible. At v2 depth most genes have 0/1 counts and every count model looks alike. (E0.1, *new*)

### Gene regulation and perturbation
- A Cas9 knockout destroys the protein, not the mRNA, so the knocked-out gene's own mRNA barely changes. Knockdown QC needs CRISPRi data or a protein or indel readout. (E0.2, *new*)
- Annotated TF → target edges (CollecTRI) are pooled across cell types and often do not hold in K562. ETS1 is not expressed in K562 at all. (E0.2, *new*)
- Co-expression is symmetric, so it cannot tell a regulator from its target: P(Y|X) ≠ P(Y|do(X)). (E0.4, *new*)

### Immunology and interferon signalling
- The type-I IFN (ISG) response happens in every cell type but with different strength: monocytes respond hardest. The shared ISG program explains 70–84% of every type's response. (E1.1, E4.5, *new* → *confirmed*)
- In Kang 2018, donors were multiplexed in the same lanes, so differences between donors are biology, not batch. (E1.1, *new*)

### Chromatin
- Housekeeping promoters stay open regardless of expression level, while lineage genes open and close with cell identity. Most nearby peaks do not regulate the gene. (E3.1, *new*)

### Tissue and tumor microenvironment (TME)
- IFN signalling is paracrine and short-range, and immune-excluded tumors produce border effects. Segmentation spillover is a real confound. (E5.1, *new*, synthetic)
- A Visium spot is a mixture of cells, and most variance sits within patients. H&E recovers composition-type biomarkers, not hidden cell-state programs. (E6.2, *new*, synthetic)

## ML and evaluation

- Compare likelihoods in the same space. A Gaussian on log1p data turned into a count PMF is the worst and least stable model. (E0.1, *new*)
- Check that nuisance parameters converge: θ stuck at its init would have understated NB. (E0.1, *new*)
- At ~10⁷ test entries, pre-register an equivalence margin; "CI includes 0" alone is too strict. (E0.1, *new*)
- Modelling cell state (latent vs floor) matters 8–10× more than NB vs Poisson. (E0.1, *new*)
- Raw Δ Pearson rewards the mean shift. Use Systema or projected residuals and shuffled controls. (E0.4, E4.5, *confirmed*)
- Simple baselines (ridge, PCA, cell-type mean) are hard to beat: ridge beat the MLP in E0.4, and PCA beat scGPT-mini in E2.1. (E0.4, E2.1, E5.1, *confirmed*)
- A random-init transformer is an essential negative control, and a good pretraining loss does not mean a good embedding. (E2.1, *new*)
- What you condition on decides what you erase: conditioning on donor removed part of the disease signal. (E1.1, *new*)
- "ATAC helps" means nothing without a shuffled-feature control. (E3.1, *new*)
- Good means do not imply good distributions. Energy distance catches a decoder that smooths cells. (E4.5, *new*)
- SAEs: Matryoshka nesting costs ~2 R² points. Rare features need a minimum count and a null outcome. Odds ratios are non-collapsible. (E2.3a, *new*)
- Under patient-grouped CV a constant baseline is anti-correlated, not neutral. CLIP + kNN compresses differences between patients. (E6.2, *new*)
- Spatial block CV is required. What the model can explain is bounded by its receptive field. (E5.1, *new*)

## Open domains (next wave)

| Domain | Experiment |
|---|---|
| Human genetics in RNA and privacy | E1.3 |
| Orthology and cross-species transfer | E2.4 |
| Haematopoiesis and cross-modal alignment | E3.2 |
| Pharmacology and dose–response | E4.3 |
| Niches shared across organs | E5.3 |
| Experimental design and closed loop | E6.3 |
