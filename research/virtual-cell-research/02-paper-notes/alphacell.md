# AlphaCell: latent manifold + optimal-transport flow matching world model

> Chuai, Chen et al. *Towards building a World Model to simulate perturbation-induced cellular dynamics*. bioRxiv (2026). DOI: 10.64898/2026.03.02.709176 · Code: not found

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed (abstract-only)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: Latent Manifold Rectification over the full protein-coding transcriptome to a "Virtual Cell Space", a large decoder back to genome-wide expression, and OT conditional flow matching with perturbations as continuous deterministic vector fields **(paper)**.
- **Why it matters for the virtual cell**: perturbation as a vector field in latent space is a natural substrate for rollouts.
- **Main claim**: robust compositional generalization and zero-shot prediction in unseen cellular contexts **(paper, no metrics in abstract)**.

## 2. Problem it solves

Prior models have incomplete latent representations, poor genome-wide reconstruction and limited cross-context generalization **(paper)**.

## 3. Method

Three components above; training details not retrieved.

## 4. Data

Not retrieved.

## 5. Evaluation and results

No metrics, baselines or datasets visible. Needs full text.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): "deterministic" vector fields cannot represent response heterogeneity; "simulate dynamics" claims need time-resolved data, which Perturb-seq endpoints are not.
- What would change my mind: comparison against linear/mean baselines under combination-held-out splits.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md): candidate dynamics module; contrasts with stochastic generators.
- Hypotheses worth testing:
  - **H-a (E6.4)**: deterministic OT-flow rollouts lose distributional spread over k steps relative to stochastic models (E-distance and variance ratio). Null: equal.

## 8. Related work

Chreode, Lingshu-Cell, CellOT, CFM.

## Resources

- [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.03.02.709176v1) · [ScienceCast](https://bio.sciencecast.org/casts/pr410naml985)
