# GEARS: knowledge-graph GNN for unseen and combinatorial genetic perturbations

> Roohani Y, Huang K, Leskovec J. *Predicting transcriptional outcomes of novel multigene perturbations with GEARS*. Nature Biotechnology 42(6), 927–935 (2024; online 2023). DOI: [10.1038/s41587-023-01905-6](https://doi.org/10.1038/s41587-023-01905-6) · Code: [snap-stanford/GEARS](https://github.com/snap-stanford/GEARS)
>
> PDF not committed (preprint bioRxiv 2022.07.12.499735).

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed (abstract-level; PMC blocked automated fetch; method details marked (uncertain) are from general knowledge)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a GNN that embeds genes using a gene co-expression graph and embeds perturbations using a Gene Ontology graph, so a never-perturbed gene inherits information from GO neighbours; combined perturbation embeddings shift the control cell's gene embeddings and a decoder predicts post-perturbation expression. **(paper; graph details (uncertain))**
- **Why it matters for the virtual cell**: the reference baseline for unseen genetic perturbations; its "simulation" split (unseen single genes, 0/1/2 seen genes in a combination) is the de facto benchmark every scFM perturbation claim is compared against. **(my take)**
- **Main claim**: outperforms prior methods on one- and two-gene perturbations across seven datasets; 40% higher precision in classifying genetic-interaction subtypes in a combinatorial screen; identifies strongest interactions twice as well as prior methods. **(paper)**

## 2. Problem it solves

Predict outcomes of perturbations (and combinations) never experimentally measured, including non-additive interactions like synergy. **(paper)**

## 3. Method

GO-graph GNN for perturbation embeddings, co-expression GNN for gene embeddings, additive composition then non-linear cross-gene layer; loss on post-perturbation expression plus a direction-of-change term. **(paper/(uncertain))**

## 4. Data

Norman 2019 (K562 CRISPRa combinations), Replogle 2022 (K562/RPE1 genome-wide CRISPRi), Adamson 2016, Dixit 2016 and others (seven total). **(paper; exact list (uncertain))**

## 5. Evaluation and results

MSE on top-20 DE genes, direction accuracy, GI-type precision. Baselines: CPA, GRN-based, no-perturbation. **(paper/(uncertain))** Later work (e.g., Ahlmann-Eltze et al., PertEval-scFM) shows simple additive/mean baselines match or beat GEARS on several of these tasks. **(my take)**

## 6. Critical analysis

- Limitations (paper): abstract-only.
- Weaknesses I see (my take): GO graph encodes prior knowledge biased to well-studied genes, so "unseen" genes with poor annotation get little signal. K562-heavy data means context generalisation is untested. Top-20-DE metrics reward predicting the common response shared by all perturbations (Systema critique).
- What would change my mind: GEARS beating a perturbation-mean baseline after Systema correction on unseen genes in RPE1 when trained on K562.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: GEARS is RNA-only with a *static* gene graph, so it cannot represent cell-type-specific regulation. The Master's H predicts that replacing/augmenting GEARS's graph with ATAC-derived, cell-type-specific peak-gene links reduces OOD error, exactly E4.1/E4.2. It is also the E0.4 "minimal virtual cell" yardstick. **(my take)**
- Hypotheses worth testing:
  - **H-gears1** (E4.1/E4.2): GEARS with a cell-type-specific gene graph built from multiome peak-gene links beats GEARS with the co-expression graph on unseen knockouts, with gains concentrated on TF perturbations. *Null*: no difference, or gains not concentrated.
  - **H-gears2** (E4.4): GEARS's advantage over the perturbation-mean baseline vanishes after Systema correction. *Null*: advantage survives.

## 8. Related work

[CPA](cpa.md), [scGen](scgen.md), [scGPT](scgpt.md), [PertFormer](pertformer.md), [Perturb-seq (Dixit 2016)](perturb-seq-dixit-2016.md), PertEval-scFM, Systema.

## Resources

- Paper: https://doi.org/10.1038/s41587-023-01905-6
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC11180609/
- Code: https://github.com/snap-stanford/GEARS
