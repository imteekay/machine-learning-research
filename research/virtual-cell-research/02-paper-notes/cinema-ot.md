# CINEMA-OT: separating confounders from treatment effects with ICA + optimal transport

> Dong M, Wang B, Wei J, de O Fonseca AH, Perry CJ, Frey A, Ouerghi F, Foxman EF, Ishizuka JJ, Dhodapkar RM, van Dijk D. *Causal identification of single-cell experimental perturbation effects with CINEMA-OT*. Nature Methods 20(11), 1769–1779 (2023). DOI: [10.1038/s41592-023-02040-5](https://doi.org/10.1038/s41592-023-02040-5) · Code: [vandijklab/CINEMA-OT](https://github.com/vandijklab/CINEMA-OT) · Data: [Dryad](https://doi.org/10.5061/dryad.4xgxd25g1)
>
> PDF not committed (open access on PMC: PMC10630139).

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read (PMC full text, main text; last ~34k characters not read)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: decompose expression with ICA into confounder components (shared across conditions) and treatment-associated components; match each treated cell to its counterfactual control cells with entropic OT on the confounder components only; per-cell treatment effect = difference. **(paper)**
- **Why it matters for the virtual cell**: it produces *per-cell* counterfactual pairs, i.e. pseudo-labels for "what would this cell look like untreated", which is the supervision a perturbation model needs and the cleanest way to study response heterogeneity (E4.3). **(my take)**
- **Main claim**: outperforms other single-cell perturbation analysis methods on simulated and real benchmarks; reveals new biology in rhinovirus/cigarette-smoke airway organoids and combinatorial cytokine stimulation of immune cells. **(paper)**

## Research summary

- **Context (already known)**: Treated and control populations differ by confounders such as cell state and composition, so mean or cluster-level differences mix confounding with treatment effect. **(paper)**
- **Gap (not known)**: How to estimate per-cell causal treatment effects from single-cell data while separating confounding variation. **(paper)**
- **Hypothesis / what they try to determine**: That ICA can isolate treatment-independent confounder components, so optimal-transport matching on them gives valid counterfactuals. **(paper)**
- **Approach (how they investigated)**: ICA components scored for treatment dependence (XICOR), entropic OT matching in confounder space, and per-cell effects as differences; tested on simulations, sci-Plex4, Alzheimer's cortex and two new datasets (airway organoids, cytokine PBMC). **(paper)**
- **Results (what they found)**: It was best on batch mixing and attribution against Mixscape, scGen, CPA, ContrastiveVI and CellOT in simulations (exact values only in figures), ran about 1 minute for 50k cells, and found rhinovirus/smoke interferon attenuation and synergistic responses. **(paper)**
- **Implications (what it means)**: It gives per-cell counterfactual pairs, but it assumes independence of confounders and fails under strong abundance imbalance. **(my take)**

## 2. Problem it solves

Treated and control populations differ by confounders (cell cycle, state, composition); naive mean differences or cluster-level comparisons mix confounding and treatment effects. **(paper)**

## 3. Method

PCA then FastICA; each component's dependence on treatment is scored with Chatterjee's coefficient (XICOR) and confounder components are those below a threshold (0.05-0.75 in the study); entropic OT (Sinkhorn-Knopp) in confounder space gives the coupling; the individual treatment effect is each cell minus its match. CINEMA-OT-W reweights cells (kNN alignment plus Leiden) for differential abundance. Synergy and attribution analyses extend it. **(paper)**

## 4. Data

Scsim simulations (about 1,000 genes x 5,000 cells); Alzheimer's entorhinal cortex (GSE138852); sci-Plex4 (GSE139944; A549/MCF7, 17 drugs); a new airway organoid set (26,420 cells; mock, RV, CSE, RVCSE); a new PBMC cytokine set (IFN-a2, b, g, l, IL-6, TNF, alone and combined, 3 donors). Data on Dryad. **(paper)**

## 5. Evaluation and results

**(paper)** Baselines: Mixscape, Harmony-Mixscape, full OT, scGen, CPA, ContrastiveVI, CellOT. Metrics: batch mixing (ASW, PCR), ARI for differential-response clusters, attribution accuracy. CINEMA-OT was best on batch mixing and attribution in balanced simulations, with the largest attribution advantage under differential abundance; exact values are in figures not visible in the text. CINEMA-OT is better at mild imbalance (ratio >= 0.75) and CINEMA-OT-W at strong imbalance (<= 0.5). About 1 minute and under 12 GB for 50,000 cells. Biology: rhinovirus broadly induced ISGs; RV plus smoke attenuated interferon response; pre-ciliated cells showed synergistic secretory-gene induction; monocytes showed the strongest synergistic chemokine response.

## 6. Critical analysis

- Limitations (paper): differential abundance violates the independence assumption; no extrapolation to unmeasured perturbation-cell pairs; not suited to abundance questions; OT oversmoothing gives false positives, undersmoothing hurts stability; no count modeling (batch effects can affect DE); dense matching matrix uses much memory; ott-jax solver is experimental.
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
