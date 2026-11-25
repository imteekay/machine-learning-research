# SpatialProp: propagating single-cell perturbations through tissue with a GNN

> Sun, Buendia, Brunet & Zou. *SpatialProp: tissue perturbation modeling with spatially resolved single-cell transcriptomics*. bioRxiv (2025). DOI: 10.64898/2025.11.30.691355 · Code: [abuendia/spatial-prop](https://github.com/abuendia/spatial-prop), [sunericd/CausalInteractionBench](https://github.com/sunericd/CausalInteractionBench)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: read (PMC full text; figures not inspected)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a 2-hop graph isomorphism network that predicts a masked cell's expression from its microenvironment; to simulate a perturbation, edit a cell and re-predict its neighbors **(paper)**.
- **Why it matters for the virtual cell**: it is the closest published attempt at the PhD's "bystander effect" module, trained only on observational spatial data.
- **Main claim**: beats a global-mean baseline, but is only *similar* to a k-hop microenvironment mean; shows causal-direction enrichment on a new benchmark, CausalInteractionBench **(paper)**.

## 2. Problem it solves

Perturb-seq perturbs cells in isolation; real tissue responses spread via signaling. Spatial perturbation data is scarce, so the paper asks how far observational spatial data can go **(paper)**.

## 3. Method

- Delaunay graph, edges >200 um pruned; GIN depth k=2 predicts the center cell from neighbors **(paper)**.
- "SparseRenorm": keep only extreme sparse changes, renormalize per cell **(paper)**.
- Weighted MAE (nonzero entries weight 10), Adam 1e-4, 50 epochs; one model per dataset; more complex GNNs did not help **(paper)**.

## 4. Data

Nine spatial datasets, mostly mouse brain (MERFISH, ISS, STARmap) plus one human heart; 140 to ~2,766 genes; ~72k to ~1.45M cells; held-out whole sections **(paper)**. CausalInteractionBench: 133 GO-derived sender-receiver terms **(paper)**. New datasets for [datasets](../03-data/datasets.md): CausalInteractionBench.

## 5. Evaluation and results

- Tissue steering: Spearman/MAE of moving a microenvironment toward a target; GNN beats global mean, ~ties k-hop mean **(paper)**.
- CausalInteractionBench: forward vs reverse response correlation difference 0.094 mean Spearman; positive Causality Enrichment Score for the model, neutral for baselines **(paper)**.
- Baselines (global mean, k-hop mean) are appropriately simple, which makes the honest tie informative **(my take)**.

## 6. Critical analysis

- Limitations (paper): per-dataset training; observational not perturbation data; GO interactions are not cell-type-specific; modest gains over k-hop mean.
- Weaknesses I see (my take): a neighbor-conditioned predictor learns correlation (co-localization, shared niche), not intervention; the "forward > reverse" asymmetry is weak evidence of causality. No time dimension.
- What would change my mind: forward-vs-reverse asymmetry surviving on real spatial Perturb-seq / Perturb-map ground truth.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md): direct precedent for the spatial module and the "propagation" hypothesis; its tie with the k-hop mean is the bar to beat.
- Hypotheses worth testing:
  - **H-a (E5.1/E5.2)**: on Perturb-map lesions, a k-hop mean baseline matches SpatialProp on first-ring neighbors; a model with directional/signaling kernels beats both. Null: no gain.
  - **H-b (E5.2)**: SpatialProp's CES gain vanishes when evaluated on held-out *tissue types* (E5.3), indicating niche memorization.

## 8. Related work

CONCERT (GP-VAE spatial kernels), Celcomen, MintFlow, Cellina; Perturb-map for ground truth.

## Resources

- [bioRxiv](https://www.biorxiv.org/content/10.64898/2025.11.30.691355v1) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12822716/)
