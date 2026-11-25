# Interpretation, extrapolation and perturbation of single cells: a unifying ontology of causal and representation-learning methods

> Dimitrov D, Schrod S, Rohbeck M, Stegle O. *Interpretation, extrapolation and perturbation of single cells*. Nature Reviews Genetics (2026). DOI: [10.1038/s41576-025-00920-4](https://doi.org/10.1038/s41576-025-00920-4) · Code: online companion resource (link in paper; not checked)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 0](../00-plan/roadmap.md#phase-0-foundations), [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a review that connects single-cell ML methods for finding dependencies and extrapolating perturbation effects, grouped by modelling concept: representation learning, causal inference, mechanistic discovery, disentanglement, population tracing. **(paper)**
- **Why it matters for the virtual cell**: it is the conceptual frame for Phase 4: what "extrapolation" to unseen perturbations or contexts means and which assumptions each method family makes. **(my take)**
- **Main claim**: single-cell analysis has moved from atlasing to causal and mechanistic inference; a unifying ontology helps practitioners pick methods for a question. **(paper)**

## 2. Problem it solves

Many methods, overlapping goals, and unstated assumptions. Practitioners cannot easily tell which model fits "predict an unseen knockout" vs "find a regulatory mechanism" vs "trace a population". **(paper)**

## 3. Method

Abstract-only (Nature paywall). From the abstract: methods are organized by modelling concept, assumptions and downstream task, with an ontology and an online resource with technical details. The review also names promising directions and under-used data properties. **(paper)** The specific ontology and directions were not checked.

## 4. Data

No new dataset.

## 5. Evaluation and results

Review; abstract-only.

## 6. Critical analysis

- Limitations (paper): abstract-only.
- Weaknesses I see (my take): taxonomies help choose methods but do not say which assumptions *hold* in a given dataset. The useful next step is tests of assumptions (e.g. is the latent effect additive across cell types?).
- What would change my mind: n/a until the full text is read.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: "extrapolation" is the core of both hypotheses. The Master's claim is that ATAC context reduces extrapolation error to unseen knockouts and contexts (E4.1); the PhD claim is extrapolation to unseen niches (E5.3).
- Hypotheses worth testing:
  - **H-EXT-1 (additivity)**: the scGen-style assumption that a perturbation is a shared latent shift holds for some cell-type pairs and fails for others; failure correlates with chromatin-accessibility distance between types. Uses E4.5 + ATAC, and is a mechanism for why E4.1 might work.
  - **H-EXT-2 (split difficulty)**: error grows with distance between test and train contexts (in expression or ATAC space). If error is flat with distance, the model is predicting a generic response (check with E4.4).

## 8. Related work

- [scGen](scgen.md), [Perturbation mini-review](perturbation-modelling-mini-review.md), [Predict, Explain, Discover](virtual-cells-predict-explain-discover.md).

## Resources

- DOI: https://doi.org/10.1038/s41576-025-00920-4
