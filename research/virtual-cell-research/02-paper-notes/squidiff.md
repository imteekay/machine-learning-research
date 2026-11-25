# Squidiff: conditional diffusion for development and perturbation

> Leong KW, Azizi E, Zou J labs (author list not verified). *Squidiff: predicting cellular development and responses to perturbations using a diffusion model*. Nature Methods (2025/2026). DOI: [10.1038/s41592-025-02877-y](https://doi.org/10.1038/s41592-025-02877-y) · Preprint: bioRxiv 10.1101/2024.11.16.623974 · Code: PyPI `Squidiff`

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. This note is **abstract-only**: the full text was not retrieved (paywall / rate limit), so no claim below comes from the methods or results sections.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: semantic encoder + conditional DDIM that generates transcriptomes under stimuli vectors (differentiation cues, gene perturbation, drugs). **(paper)**
- **Why it matters for the virtual cell**: extends perturbation modeling to developmental trajectories and transient states; relevant to the PhD tissue/organoid plan. **(my take)**
- **Main claim**: predicts iPSC differentiation to three germ layers, captures transient states others miss; applied to blood-vessel organoids and responses to neutron irradiation and growth factors. **(paper)**

## 2. Problem it solves

Predicting transcriptomic responses across cell types to diverse environmental changes. **(paper)**

## 3. Method

Semantic encoder gives a latent; stimuli are vectors added/interpolated in semantic space; DDIM decodes. Details abstract-only. **(paper)**

## 4. Data

iPSC differentiation, blood-vessel organoids, irradiation, growth-factor and gene/drug perturbation data (abstract-level). **(paper)**

## 5. Evaluation and results

no numbers recorded (abstract-only); baselines unknown.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): semantic-vector arithmetic resembles scGen's latent δ; risk of the same linear-shift limitation. **(my take)**
- What would change my mind: good performance on unseen cell types where gene-space shift fails.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: relevant to E4.5 (latent arithmetic) and the PhD spatial/tissue hypothesis. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: on Kang IFN-β (E4.5 split), Squidiff beats mini scGen on residual Pearson; falsified if not. (2) *H*: its transient-state advantage disappears versus a trajectory-interpolation baseline.

## 8. Related work

scGen ([note](scgen.md)), PerturbDiff ([note](perturbdiff.md)), CellFlow.

## Resources

- bioRxiv: https://www.biorxiv.org/content/10.1101/2024.11.16.623974
