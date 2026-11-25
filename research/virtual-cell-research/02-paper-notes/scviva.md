# scVIVA: a VAE that models a cell together with its spatial environment

> Levy et al. (Yosef lab; full list not re-checked). *scVIVA: a probabilistic framework for representation of cells and their environments in spatial transcriptomics*. bioRxiv (2025). DOI: 10.1101/2025.06.01.657182 · Code: [scvi-tools docs](https://docs.scvi-tools.org/en/stable/user_guide/models/scviva.html) (moved to scVIVA-Tools from scvi-tools v1.5) · [Video](https://www.youtube.com/watch?v=ykVoYVyCVd0)
>
> bioRxiv rate-limited; built from abstract + scvi-tools docs. **Abstract-only** for numbers.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a scVI-family deep generative model whose stochastic cell embedding is informed by the cell's own expression and its neighbors' expression; outputs embeddings and normalized expression **(paper)**.
- **Why it matters for the virtual cell**: it is the probabilistic, count-aware baseline for niche-aware representation; small, principled, and uses proper NB likelihoods — the kind of baseline large spatial FMs must beat.
- **Main claim**: finds a spatially restricted tumor-promoting endothelial population in breast cancer and niche-associated T-cell states shared across cancers **(paper)**.

## 2. Problem it solves

Most cell embeddings ignore neighbors, so cells with identical intrinsic programs in different niches are merged **(paper)**.

## 3. Method

VAE with count likelihood; neighborhood expression/composition used in the generative/inference model; single-cell-resolution data required (not Visium/Slide-seq); GPU recommended; scales >1M cells **(paper/docs)**. Exact neighbor term: abstract-only.

## 4. Data

Breast cancer and multiple cancer single-cell spatial datasets (abstract-only for names).

## 5. Evaluation and results

Abstract-only; discovery case studies; differential expression accounting for niche.

## 6. Critical analysis

- Limitations (paper/docs): needs single-cell resolution.
- Weaknesses I see (my take): segmentation errors leak neighbor transcripts into a cell, which a neighbor-aware model can mistake for niche biology.
- What would change my mind: niche states persisting after segmentation-contamination correction (e.g. comparing with nuclear-only counts).

## 7. Relevance to my research

- **Hypotheses**: strong cheap baseline for E5.1 (niche adds information beyond cell type) and for PhD H; the probabilistic generative module in the proposal fits this family.
- Hypotheses worth testing:
  - **H-a (E5.1)**: scVIVA latent explains more variance in held-out neighbor composition than scVI latent of the same size; falsified if gap ≈ 0.
  - **H-b (E5.2)**: a scVIVA-style conditional VAE with a perturbation covariate matches large spatial FMs on Perturb-map bystander prediction (E-distance).
  - **H-c**: some niche-specific states vanish after segmentation-contamination correction.

## 8. Related work

[NicheCompass](nichecompass.md), [Nicheformer](nicheformer.md), scVI, resolVI.

## Resources

- bioRxiv: https://www.biorxiv.org/content/10.1101/2025.06.01.657182v1.full.pdf
- Docs: https://docs.scvi-tools.org/en/stable/user_guide/models/scviva.html
- Video: https://www.youtube.com/watch?v=ykVoYVyCVd0
