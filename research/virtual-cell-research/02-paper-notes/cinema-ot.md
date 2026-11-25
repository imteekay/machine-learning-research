# CINEMA-OT: separating confounders from treatment effects with ICA + optimal transport

> Dong M, Wang B, Wei J, de O Fonseca AH, Perry CJ, Frey A, Ouerghi F, Foxman EF, Ishizuka JJ, Dhodapkar RM, van Dijk D. *Causal identification of single-cell experimental perturbation effects with CINEMA-OT*. Nature Methods 20(11), 1769–1779 (2023). DOI: [10.1038/s41592-023-02040-5](https://doi.org/10.1038/s41592-023-02040-5) · Code: [vandijklab/CINEMA-OT](https://github.com/vandijklab/CINEMA-OT) · Data: [Dryad](https://doi.org/10.5061/dryad.4xgxd25g1)
>
> PDF not committed (open access on PMC: PMC10630139).

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed (abstract-only; PMC blocked automated fetch)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: decompose expression with ICA into confounder components (shared across conditions) and treatment-associated components; match each treated cell to its counterfactual control cells with entropic OT on the confounder components only; per-cell treatment effect = difference. **(paper)**
- **Why it matters for the virtual cell**: it produces *per-cell* counterfactual pairs, i.e. pseudo-labels for "what would this cell look like untreated", which is the supervision a perturbation model needs and the cleanest way to study response heterogeneity (E4.3). **(my take)**
- **Main claim**: outperforms other single-cell perturbation analysis methods on simulated and real benchmarks; reveals new biology in rhinovirus/cigarette-smoke airway organoids and combinatorial cytokine stimulation of immune cells. **(paper)**

## 2. Problem it solves

Treated and control populations differ by confounders (cell cycle, state, composition); naive mean differences or cluster-level comparisons mix confounding and treatment effects. **(paper)**

## 3. Method

ICA on preprocessed expression; a test identifies components independent of treatment (confounders); entropic OT on those components gives a coupling; the treatment-effect matrix supports response clustering and synergy analysis. A reweighted variant handles imbalanced cell compositions. **(paper; variant naming (uncertain))**

## 4. Data

Simulations plus two new datasets: airway organoids (rhinovirus, cigarette smoke) and combinatorial cytokine stimulation of immune cells (on Dryad). **(paper)** Possibly new datasets for the registry.

## 5. Evaluation and results

Benchmarks vs. other perturbation-analysis/matching methods on simulated and real data; numbers abstract-only.

## 6. Critical analysis

- Limitations (paper): abstract-only.
- Weaknesses I see (my take): identification relies on the assumption that confounders are captured by treatment-independent ICA components; a treatment that shifts cell cycle would be partially absorbed as "confounder". Linear ICA may miss non-linear confounding.
- What would change my mind: recovering a known ground-truth per-cell effect in a spike-in or barcoded-lineage design where counterfactual pairs are partly observable.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: the Master's E-distance metric compares distributions; CINEMA-OT gives per-cell matches so I can ask *which* cells respond. E3.3 needs per-cell response to regress on baseline motif accessibility; CINEMA-OT matching is a principled way to get it. **(my take)**
- Hypotheses worth testing:
  - **H-cin1** (E3.3): per-cell CINEMA-OT effects for a TF knockout correlate with the cell's baseline motif accessibility (from matched control ATAC) more than effects from random matching. *Null*: equal partial R².
  - **H-cin2** (E4.3): training a distributional model on CINEMA-OT pairs instead of unpaired distributions improves responder/non-responder recovery (bimodality of predicted effect). *Null*: no change.

## 8. Related work

CellOT, Mixscape, [Labeled GW-OT](labeled-gw-ot-cross-modality-perturbation.md), [CPA](cpa.md), causal inference matching.

## Resources

- Paper: https://doi.org/10.1038/s41592-023-02040-5
- PMC: https://pmc.ncbi.nlm.nih.gov/articles/PMC10630139/
- Data: https://doi.org/10.5061/dryad.4xgxd25g1
