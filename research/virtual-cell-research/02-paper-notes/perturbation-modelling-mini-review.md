# Perturbation modelling across single-cell omics (mini-review): a short map of technologies and model families

> Gavriilidis GI, Vasileiou V, Orfanou A, Ishaque N, Psomopoulos F. *A mini-review on perturbation modelling across single-cell omic modalities*. Computational and Structural Biotechnology Journal (2024). DOI: [10.1016/j.csbj.2024.04.058](https://doi.org/10.1016/j.csbj.2024.04.058) · PMC: [PMC11076269](https://pmc.ncbi.nlm.nih.gov/articles/PMC11076269/)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 0](../00-plan/roadmap.md#phase-0-foundations), [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read (Europe PMC full text; last ~46k characters, mostly references, not read)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a short review of perturbation-modelling goals, single-cell perturbation technologies (CRISPR, compounds) across omic modalities, and computational methods. **(paper)**
- **Why it matters for the virtual cell**: a quick orientation to the landscape before reading individual Phase 4 models. **(my take)**
- **Main claim**: methods range from classical statistics to shallow ML, autoencoders, GRN-informed models, ensembles and LLM-inspired FMs; key open problems are benchmarking, interoperability, causality and multi-omic / spatial data. **(paper)**

## Research summary

- **Context (already known)**: Many single-cell perturbation technologies, datasets and models exist across omic modalities. **(paper)**
- **Gap (not known)**: No standard benchmarks or best practices exist, so it is hard to know which method families work. **(paper)**
- **Hypothesis / what they try to determine**: What are the objectives, tools and open problems of perturbation modelling across modalities. **(paper)**
- **Approach (how they investigated)**: Literature review: technologies and databases, then 41 tools clustered into six groups (UMAP, k-means), benchmarking evidence and outlook. **(paper)**
- **Results (what they found)**: Methods span statistics, ML, GRN-based, deep generative and foundation models; open problems are data scarcity, benchmarking gaps, interpretability, interoperability and translation. **(paper)**
- **Implications (what it means)**: A map of the landscape for orientation; it contains no new results and predates the finding that simple baselines rival deep models. **(my take)**

## 2. Problem it solves

The number of perturbation tools and datasets is growing fast and is hard to follow for both biologists and computational scientists. **(paper)**

## 3. Method

Review structure (from abstract) **(paper)**:
1. Objectives of perturbation modelling (effects on TFs, signal transducers, pathways, cell states).
2. Perturbation technologies: genetic (CRISPR) and chemical, across omic modalities.
3. Method families: statistical inference; shallow ML and autoencoders (e.g. scGen-type); GRN-informed models; combinatorial / ensemble approaches; large FMs.
4. Challenges and directions: perturbation atlases, multi-omic and spatial data, causal ML for interpretability, multi-task learning, interoperability and benchmarking pitfalls.

Per the full text: 41 tools clustered into six method groups (shallow/statistical, mixed ML, similarity, GRN-based, deep generative, foundation models) by UMAP and k-means. **(paper)**

## 4. Data

No new dataset; surveys Perturb-seq-family, sci-Plex, Perturb-CITE-seq and Perturb-ATAC assays, and databases scPerturb (44 harmonised datasets, E-distance) and PertPy. **(paper)**

## 5. Evaluation and results

Review; no new results. It relays small-scale comparisons (e.g. CellOracle over GRN baselines, scVIDR over scGEN/CellOT on R2) and notes the lack of standard benchmarks. **(paper)**

## 6. Critical analysis

- Limitations (paper): data scarcity (static snapshots, few spatial/longitudinal sets), small benchmarks, interpretability, limited FM zero-shot performance, interoperability, limited clinical translation.
- Weaknesses I see (my take): from April 2024, before the main negative baseline results (linear / mean models matching deep models on Perturb-seq) and before most flow-matching models. Use as a map, not as a ranking.
- What would change my mind: n/a.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: its call for multi-omic perturbation data and causal ML is exactly the Master's setup (Perturb-multiome, E3.3, E4.1). Its spatial direction is the PhD setup (Perturb-map, E5.2).
- Hypotheses worth testing:
  - **H-MR-1 (GRN prior vs ATAC)**: a GRN-informed model whose prior comes from cell-type-specific ATAC peak-gene links beats the same model with a generic (non-cell-type) GRN prior on unseen knockouts. This separates "any network prior helps" from "context-specific chromatin helps" (E4.1/E4.2).

## 8. Related work

- [Interpretation, extrapolation and perturbation](interpretation-extrapolation-perturbation-review.md) (deeper, more recent), [scGen](scgen.md), [Perturb-seq (Dixit 2016)](perturb-seq-dixit-2016.md).

## Resources

- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11076269/
