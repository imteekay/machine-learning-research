# CellOS: a 12B multi-view cell foundation model with LLM-JEPA alignment

> *CellOS: Learning a World Model of Cellular State through Joint Embedding Prediction*. bioRxiv (2026). DOI: 10.64898/2026.06.18.733163 · Code: not found

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed (abstract via Sciety summary; bioRxiv 429)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a multi-view model learning cell states from paired expression and "perception" views, 12B parameters, trained on 390.5M transcriptomes with three stages: causal cell-sentence LM, dense-to-MoE expansion, LLM-JEPA latent alignment **(paper, abstract)**.
- **Why it matters for the virtual cell**: JEPA is the leading alternative to reconstruction for world models; tests predicting in latent space.
- **Main claim**: beats SOTA single-cell FMs on cell-state annotation, batch integration and perturbation-response prediction **(paper, unreviewed)**.

## 2. Problem it solves

Single-view, reconstruction-trained FMs capture expression levels but not complementary views of state.

## 3. Method

Three-stage training above; "perception" view not defined in what I retrieved.

## 4. Data

390.5M single-cell transcriptomes **(paper)**. Sources not retrieved.

## 5. Evaluation and results

Abstract-only; no numbers or baselines list. Perturbation-response baselines (mean, linear) unknown.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): "world model" in the title with no stated rollouts; scale confounds the JEPA contribution; needs an ablation of LLM-JEPA vs plain cell-sentence LM.
- What would change my mind: the JEPA stage giving no gain over stage 1 on held-out perturbations.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md): JEPA candidate for the latent dynamics module.
- Hypotheses worth testing:
  - **H-a (E6.4)**: latent rollouts keep lower E-distance at horizon k than recursive one-step generation on time-course perturbation data. Null: equal.
  - **H-b**: removing LLM-JEPA alignment does not change perturbation-response error (scale, not objective, drives gains).

## 8. Related work

Cell-JEPA, Chreode, Lingshu-Cell, scGPT/Geneformer.

## Resources

- [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.06.18.733163v2) · [Sciety](https://sciety.org/articles/activity/10.64898/2026.06.18.733163)
