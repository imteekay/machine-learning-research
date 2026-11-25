# NicheCompass: interpretable niche embeddings from cell-communication-informed graph learning

> Birk, Bonafonte-Pardàs, Feriz, et al., Lotfollahi (senior; author list not fully re-checked). *Large-scale characterization of cell niches in spatial atlases using bio-inspired graph learning*. bioRxiv (2024); published as *Quantitative characterization of cell niches in spatially resolved omics data*, Nature Genetics (2025). DOI: 10.1101/2024.02.21.581428 · Code: [Lotfollahi-lab/nichecompass](https://github.com/Lotfollahi-lab/nichecompass)
>
> bioRxiv rate-limited; built from search snippets of preprint/PMC. **Abstract-only** for numbers.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a GNN autoencoder over a spatial neighbor graph whose latent dimensions are tied to prior **gene programs** (ligand-receptor, intra-cellular signaling pathways), giving interpretable niche embeddings; supports reference-atlas mapping and batch removal **(paper)**.
- **Why it matters for the virtual cell**: it explains niches by *which signals* are active, the mechanistic layer a bystander-effect model needs.
- **Main claim**: scales to atlases (8.4M-cell, 239-section mouse brain), characterizes tumor niches in breast and NSCLC, and extends to multimodal (spatial ATAC+RNA) data **(paper)**.

## Research summary

- **Context (already known)**: niche methods exist for spatial data but are often not interpretable, do not scale to multi-sample atlases and cannot map new samples to a reference. **(paper)**
- **Gap (not known)**: whether a graph model with prior gene programs can give interpretable, scalable niche embeddings. **(paper)**
- **Hypothesis / what they try to determine**: tying latent dimensions to ligand-receptor and signalling programs yields niches explained by active signals. **(paper)**
- **Approach (how they investigated)**: a GNN autoencoder on a spatial neighbour graph with prior-program-constrained latents and scArches-style conditional embeddings; applied to mouse embryo, breast cancer, NSCLC and an 8.4M-cell mouse brain atlas. **(paper)**
- **Results (what they found)**: abstract-only: scales to 239-section atlas, separates basal vs luminal tumor niches and characterises NSCLC TME; no benchmark numbers read. **(paper)**
- **Implications (what it means)**: niche identity can be explained by pathway activity, but discovery is limited to known programs and L-R co-expression is not proof of signalling. **(my take)**

## 2. Problem it solves

Niche identification methods are not interpretable, don't scale to multi-sample atlases, and can't query new samples against a reference **(paper)**.

## 3. Method

Spatial kNN/radius graph; GNN encoder; decoders reconstruct own and neighbor expression with latent factors constrained by prior gene programs (NicheNet/OmniPath style); conditional embeddings for batch removal (scArches-like) **(paper; details abstract-only)**.

## 4. Data

Mouse embryo, human breast cancer, NSCLC TME, whole mouse brain atlas (8.4M cells, 239 sections, 4 mice), multimodal spatial data **(paper)**.

## 5. Evaluation and results

Abstract-only for metrics. Qualitative: basal vs luminal tumor niches; NSCLC TME characterization; atlas-scale mapping.

## 6. Critical analysis

- Limitations (paper): not extracted.
- Weaknesses I see (my take): prior gene programs constrain discovery to known pathways; L-R co-expression is not proof of signaling.
- What would change my mind: program activities predicting measured responses in neighbors of perturbed cells.

## 7. Relevance to my research

- **Hypotheses**: PhD H — NicheCompass programs are a natural interpretable readout for "signal bleed"; the proposal's directional kernels could be parameterized per program **(my take)**. Multimodal extension touches Master's ATAC context.
- Hypotheses worth testing:
  - **H-a (E5.2)**: in Perturb-map, neighbors of KO patches show changes concentrated in programs whose ligand is downstream of the KO gene; falsified if changes are program-agnostic.
  - **H-b (E5.1)**: program-constrained latents predict niche labels as well as unconstrained ones (interpretability is free); falsified if >5 pt drop.

## 8. Related work

[scVIVA](scviva.md), [HEIST](heist.md), [Nicheformer](nicheformer.md), scArches, NicheNet.

## Resources

- bioRxiv: https://www.biorxiv.org/content/10.1101/2024.02.21.581428v1
- Nature Genetics version via PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11985353/
