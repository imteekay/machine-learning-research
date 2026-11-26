# Learnings ledger

Biology and ML lessons from every experiment, grouped by domain. Each experiment README has its own "Learnings ledger" section; copy its bullets here and tag each one *new*, *confirms Ex.y* or *contradicts Ex.y*.

## Biology

### Assay physics and count statistics
- Droplet (UMI) zeros are explained by depth plus NB overdispersion, with no excess dropout. ZINB gains < 0.13% of the LL. (E0.1, *new*)
- Most overdispersion of HVGs is cell-state heterogeneity: marginal θ ≈ 0.1, but θ ≈ 1 given the latent. (E0.1, *new*)
- Deeper chemistry (v3) makes overdispersion visible. At v2 depth most genes have 0/1 counts and every count model looks alike. (E0.1, *new*)

### Human genetics in RNA and privacy
- Donor identity is spread across hundreds of genes (sex, HLA, ribosomal, mitochondrial, known eQTLs). The 67 pre-registered sensitive genes alone link 28% of donors, and removing them cuts linkage only 6%. (E1.3, *new*)
- Some linkage is technical (handling and stress genes), so the figure is an upper bound on genetic identifiability. (E1.3, *new*)

### Cross-species biology
- Parenchymal and epithelial cell types (hepatocyte, acinar, goblet, endothelium) transfer from mouse to human, while immune cells barely transfer under any method. (E2.4, *new*)
- Species, atlas and assay shift cost about two thirds of the within-species F1 (0.85 → 0.28). (E2.4, *new*)
- Genes with no 1:1 ortholog still carry cell identity (18× chance with no ortholog genes at all), so protein sequence carries some conserved function. (E2.4, *new*)

### Gene regulation and perturbation
- A Cas9 knockout destroys the protein, not the mRNA, so the knocked-out gene's own mRNA barely changes. Knockdown QC needs CRISPRi data or a protein or indel readout. (E0.2, *new*)
- Annotated TF → target edges (CollecTRI) are pooled across cell types and often do not hold in K562. ETS1 is not expressed in K562 at all. (E0.2, *new*)
- Co-expression is symmetric, so it cannot tell a regulator from its target: P(Y|X) ≠ P(Y|do(X)). (E0.4, *new*)

### Pharmacology and dose-response
- In sci-Plex 3 most drug responses shift the whole population: the true spread equals the vehicle's and a true mean shift reproduces the distribution in 89% of active conditions. (E4.3, *new*)
- Subpopulation responses concentrate in epigenetic regulators (HDAC inhibitors: 62 of 86 heterogeneous conditions) and Aurora kinase inhibitors, and grow with dose (2.6% at 10 nM to 17% at 10 µM). (E4.3, *new*)
- A new dose of a known drug is easy to predict; a new drug from its MoA label alone is hard (mean-shift Pearson 0.53 vs 0.97 for the oracle). (E4.3, *new*)

### Immunology and interferon signalling
- The type-I IFN (ISG) response happens in every cell type but with different strength: monocytes respond hardest. The shared ISG program explains 70–84% of every type's response. (E1.1, E4.5, *new* → *confirmed*)
- In Kang 2018, donors were multiplexed in the same lanes, so differences between donors are biology, not batch. (E1.1, *new*)

### Chromatin
- RNA and ATAC of the same cell can be matched across unseen donors and sites (top-10 for 30% of cells among 2,000). Matching resolves state within a cell type, and is easiest for dynamic states (progenitors, activated T cells) and hardest for quiescent or terminal ones (CD16+ monocytes, naive B). (E3.2, *new*)
- Lineage priming did not make progenitors harder to match. (E3.2, *new*)
- ATAC LSI has a large site offset: a global-mean imputation has R² of −0.18 on an unseen site. (E3.2, *new*)
- Housekeeping promoters stay open regardless of expression level, while lineage genes open and close with cell identity. Most nearby peaks do not regulate the gene. (E3.1, *new*)

### Spatial niches (mouse brain)
- Smoothing each cell's expression over its 15 nearest neighbours doubles niche recovery over the cell alone (NMI 0.31 vs 0.16); coarse cell-class composition does not (0.15), so niche identity lives in fine cell states. (E5.3, *new*)
- Niche recovery is limited by the test tissue, not the training tissue: the transfer matrix is flat by row and varies 5× by column (cerebellum below 0.1, thalamus and hippocampus 0.33). Niches generalise to another animal and cutting plane. (E5.3, *new*)

### Tissue and tumor microenvironment (TME)
- IFN signalling is paracrine and short-range, and immune-excluded tumors produce border effects. Segmentation spillover is a real confound. (E5.1, *new*, synthetic)
- A Visium spot is a mixture of cells, and most variance sits within patients. H&E recovers composition-type biomarkers, not hidden cell-state programs. (E6.2, *new*, synthetic)

## ML and evaluation

- A learned niche encoder beats expression alone, an untrained network and a shuffled graph in every tissue but never beats hand-crafted neighbour smoothing (0.274 vs 0.331 NMI). A multi-tissue model did not beat one trained on a single section of the target (1 of 10 tissues). The gene basis explains at most 0.008 of the gap and the objective (SwAV vs InfoNCE) does not matter. Always report the 1- and 2-hop neighbour-mean baseline. (E5.3, *confirms E2.1*)

- Flow matching did not beat a mean-shift MLP (worse for unseen doses, +0.15; narrowly better for unseen drugs, −0.04; no gain on the 86 heterogeneous conditions). The mean-shift error dominates, so improve the mean first. A mean-oracle row separates mean error from shape error. (E4.3, *new*)
- Energy distance is dominated by the mean in high dimension: it detects a bimodal shape up to about 10 isotropic dimensions at n = 100, or about 50 with PCA-like variances, and MSE of the mean cannot see it at all (E0.3, answered in E4.3). A thresholded responder fraction cannot tell bimodal from shifted. (E4.3, *new*)
- CPA-style VAEs leak the condition into the latent unless an adversary and a tight bottleneck are used, and then collapse the spread. (E4.3, *new*)

- Contrastive RNA↔ATAC alignment beats gene-activity matching in 4 of 4 held-out sites (FOSCTTM 0.036 vs 0.118), with shuffled-pair and untrained controls at chance. Gromov–Wasserstein fails. (E3.2, *new*)
- A strict ">" top-k gives hard transport plans a free perfect score. Use tie-aware ranks. Pooled FOSCTTM is confounded by pool composition. (E3.2, *new*)
- Donors overlap across sites in the NeurIPS multiome data, so leave-one-site-out leaks donors unless the pool is restricted. (E3.2, *new*)
- The embedding that matches best does not impute best: RNA-kNN beats it in 3 of 4 sites (R² 0.587 vs 0.554). (E3.2, *new*)

- A mini UCE loses to ortholog PCA by 0.18 macro-F1 on mouse → human. Protein tokens add signal (+0.037 vs untrained, +0.079 vs random tokens) and make non-ortholog transfer possible. (E2.4, *confirms E2.1*)
- A within-species held-out set overstates cross-species skill 3× (0.85 vs 0.28). (E2.4, *new*)

- Gene removal is a weak privacy tool: dropping 1000 donor-specific genes cuts linkage 27% but costs 0.13 cell-type macro-F1. Matched-random removal is the control. (E1.3, *new*)

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
| Experimental design and closed loop | E6.3 |
