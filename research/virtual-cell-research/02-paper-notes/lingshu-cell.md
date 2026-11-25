# Lingshu-Cell: masked discrete diffusion as a cellular world model

> *Lingshu-Cell: A generative cellular world model for transcriptome modeling toward virtual cells*. arXiv (2026). arXiv: 2603.25240 · Code: not linked on abstract page

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed (abstract-only)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a masked discrete diffusion model over discrete expression tokens for ~18,000 genes, without gene selection, with joint cell-type/donor plus perturbation conditioning **(paper)**.
- **Why it matters for the virtual cell**: a distribution-level generator (not a mean predictor) that handles novel identity-perturbation combinations.
- **Main claim**: reproduces transcriptome distributions, markers and subtype proportions; reports leading results on the Virtual Cell Challenge H1 benchmark and PBMC cytokine responses (self-reported, no metrics in abstract) **(paper)**.

## 2. Problem it solves

Direct control-to-perturbed mappings ignore distributions; sparse non-sequential expression fits poorly with autoregressive tokenization.

## 3. Method

Masked discrete diffusion in a discrete token space; identity and perturbation embedded jointly **(paper)**. Details abstract-only.

## 4. Data

Multi-tissue, multi-species corpora (unnamed), VCC H1 genetic perturbation data, human PBMC cytokine data **(paper, abstract)**.

## 5. Evaluation and results

Abstract-only; no numbers. Whether mean/linear baselines were included is unknown.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): "leading" on H1 is meaningful only against the mean baseline, which is hard to beat on VCC metrics; "world model" label is for a one-step conditional generator, with no rollouts.
- What would change my mind: win over mean/linear baselines on E-distance and DE-gene metrics on held-out perturbations.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md): candidate generative module of the 3-module design (cited there).
- Hypotheses worth testing:
  - **H-a (E4.x)**: on held-out cell-type x perturbation cells, diffusion sampling beats mean-shift baselines on distributional E-distance but not on mean-level Pearson-delta.
  - **H-b (E6.4)**: applying it recursively over time compounds error like other one-step models.

## 8. Related work

scGen, CPA, STATE, Chreode, AlphaCell.

## Resources

- [arXiv](https://arxiv.org/abs/2603.25240)
