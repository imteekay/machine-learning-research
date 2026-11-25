# Flow matching in computational biology: review of flow-matching generative models from molecules to cells

> Morehead A, Atanackovic L, Hegde A, ..., Krishnapriyan A (9 authors). *Flow matching for generative modelling in bioinformatics and computational biology*. Nature Machine Intelligence 8(4) (2026). DOI: [10.1038/s42256-026-01220-0](https://doi.org/10.1038/s42256-026-01220-0) · Open access: [eScholarship](https://escholarship.org/uc/item/07b9j5th) (CC BY 4.0)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 0](../00-plan/roadmap.md#phase-0-foundations), [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a review of flow matching (FM): theory, then applications to small molecules, proteins, DNA/RNA, their interactions, and single- and multi-cell modelling, plus a list of open-source FM tools. **(paper)**
- **Why it matters for the virtual cell**: many 2025–26 perturbation models (e.g. CellFlow) are flow-matching models that map a control population to a perturbed population. Understanding FM is needed to judge distributional claims in E4.3. **(my take)**
- **Main claim**: FM is a simple, scalable training method for generative models that is spreading across computational biology. **(paper)**

## 2. Problem it solves

Mean-prediction models cannot represent heterogeneous responses (responders vs non-responders). Diffusion models handle distributions but are costly; FM learns a vector field that transports a source distribution to a target with a simple regression loss. **(my take, standard FM background)**

## 3. Method

Abstract-level only: I located the open-access PDF but could not extract its text in this session. Sections include "Fundamentals of flow matching" and application areas listed above. **(paper)** Not checked: which single-cell models are covered, and the authors' open challenges.

Background **(my take)**: conditional FM regresses a network v(x,t,c) onto the velocity of a simple path between paired source and target samples; optimal-transport couplings (OT-CFM) make paths straighter. For perturbations, source = control cells, target = perturbed cells, condition c = perturbation (and, for my work, chromatin context).

## 4. Data

No new dataset.

## 5. Evaluation and results

Review; abstract-only.

## 6. Critical analysis

- Limitations (paper): not checked (abstract-only).
- Weaknesses I see (my take): in unpaired single-cell data the OT coupling between control and perturbed cells is a modelling choice, not ground truth. A good distribution match (low E-distance) can be reached while individual cell trajectories are wrong. Also, FM models are often compared with mean baselines using MSE, which hides their advantage or lack of it.
- What would change my mind: a benchmark where FM models beat a "control + mean shift" baseline on E-distance for unseen perturbations, not only for seen ones.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: the Master's metric is E-distance between predicted and observed populations, which is a natural fit for FM. A flow conditioned on RNA+ATAC is one candidate architecture for E4.1.
- Hypotheses worth testing:
  - **H-FM-1 (distributional value, E4.3)**: on a dataset with known responder heterogeneity (e.g. variable knockdown efficiency), a conditional FM model gets lower E-distance than a mean-shift baseline, while MSE on the mean is equal. *Null*: E-distance also equal.
  - **H-FM-2 (context as condition)**: adding ATAC-derived context to the FM condition lowers E-distance on unseen cell contexts. This is E4.1 with a distributional model.
  - **H-FM-3 (E0.3 link)**: E-distance rewards mean-shift baselines more than expected when sample sizes are small; test by subsampling cells.

## 8. Related work

- [scGen](scgen.md) (deterministic latent shift, the simplest transport map), [Perturbation mini-review](perturbation-modelling-mini-review.md).

## Resources

- Open access: https://escholarship.org/uc/item/07b9j5th
- DOI: https://doi.org/10.1038/s42256-026-01220-0
