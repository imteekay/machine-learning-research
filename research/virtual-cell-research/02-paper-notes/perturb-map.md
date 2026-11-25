# Perturb-map: spatial CRISPR genomics of the tumor microenvironment

> Dhainaut et al. *Spatial CRISPR genomics identifies regulators of the tumor microenvironment*. Cell 185(7):1223-1239 (2022). DOI: 10.1016/j.cell.2022.02.015 · PMC8992964 · Code: n/a

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: skimmed (PMC blocked by captcha; abstract-level from search summaries)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: Perturb-map uses Pro-Code protein barcodes to read out CRISPR-knockout clones in situ, combined with multiplex imaging and spatial transcriptomics **(paper)**.
- **Why it matters for the virtual cell**: it is the spatial perturbation dataset in the proposal: perturbations with known location, so neighbor (bystander) responses can be measured.
- **Main claim**: dozens of genes knocked out in parallel in a mouse lung cancer model; e.g. Tgfbr2 loss gave a fibro-mucinous microenvironment with T-cell exclusion via higher TGFb bioavailability **(paper, abstract-level)**.

## Research summary

- **Context (already known)**: Pooled CRISPR screens read growth or FACS phenotypes and lose spatial context, while tumor immune composition is a spatial phenotype. **(paper)**
- **Gap (not known)**: Which genes regulate tumor microenvironment composition in vivo in a spatially resolved, multiplexed way is not known. **(paper)**
- **Hypothesis / what they try to determine**: Whether barcoded CRISPR knockouts imaged in situ can link tumor-cell genes to local immune and spatial phenotypes. **(paper)**
- **Approach (how they investigated)**: Pro-Code barcoded CRISPR pool in mouse lung cancer, with multiplex imaging, histopathology and spatial transcriptomics per clone; full text not read (PMC captcha). **(paper)**
- **Results (what they found)**: Dozens of genes knocked out in parallel; for example Tgfbr2 loss gave a fibro-mucinous microenvironment with T-cell exclusion via higher TGFb bioavailability (abstract-level). **(paper)**
- **Implications (what it means)**: It is a data and assay template for perturbations with known location, enabling neighbor-effect modeling, though per-KO replication and resolution look limiting. **(my take)**

## 2. Problem it solves

Pooled screens lose spatial context; tumor immune composition is a spatial, multicellular phenotype.

## 3. Method

Pro-Code barcoded CRISPR pool in tumor cells; barcodes imaged in tissue; per-clone growth, histopathology, immune composition and spatial transcriptomics **(paper)**. Further details not retrieved.

## 4. Data

Mouse lung-cancer lesions with dozens of KOs plus spatial transcriptomics (abstract-level). Candidate entry for [datasets](../03-data/datasets.md): Perturb-map (accessions not retrieved).

## 5. Evaluation and results

Abstract-only: per-KO effects on growth and immune composition; Tgfbr2 case study. No ML baselines (it is a data resource).

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): few lesions per KO, mouse model, low-plex transcriptomics; spot resolution may blur single-cell neighbors; clones are spatially clustered so spatial and perturbation effects are confounded.
- What would change my mind: if held-out-lesion splits show too little variance to evaluate neighbor effects, the dataset is too small for E5.2.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md): ground truth for the bystander/patch-and-border tasks.
- Hypotheses worth testing:
  - **H-a (E5.2)**: with whole lesions held out, a spatial model predicts the first-ring neighbor response better than an isolated-cell model; effect shrinks beyond ~N um. Null: no ring effect.
  - **H-b**: a SpatialProp-trained-on-observational-data model, zero-shot on Perturb-map, predicts the sign of the TGFb-axis changes better than chance.

## 8. Related work

Spatial CRISPR-seq / in-situ screens, SpatialProp, CONCERT.

## Resources

- [Cell](https://doi.org/10.1016/j.cell.2022.02.015) · [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8992964/) · [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2021.07.13.451021v1)
