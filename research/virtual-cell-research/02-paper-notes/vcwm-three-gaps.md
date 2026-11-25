# What makes a virtual cell a world model: three gaps, three axes, a roadmap

> *What Makes a Virtual Cell a World Model? Three Gaps, Three Experiments (ICML listing: Three Axes), and a Roadmap*. OpenReview / ICML (2026). [OpenReview](https://openreview.net/forum?id=YyI469JbU3) · Code: n/a

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed (abstract-only; OpenReview blocked, ICML listing and Research Square abstract via search)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: an evaluation lens, not a model: when is a "virtual cell" a world model **(paper)**.
- **Why it matters for the virtual cell**: gives falsifiable criteria so the PhD can claim (or not) a world model.
- **Main claim**: three gaps: representation is not dynamics, prediction is not intervention, multimodality is not multiscale **(paper)**. Three axes (intervenable trajectory rollout, scale coupling, biology-constrained transitions), each with an L0-L3 capability ladder **(paper)**.

## Research summary

- **Context (already known)**: "World model" is applied loosely to virtual cells, and FM embeddings and multimodal models are often called world models. **(paper)**
- **Gap (not known)**: What concretely separates a virtual cell from a world model, and how to tell. **(paper)**
- **Hypothesis / what they try to determine**: Three gaps (representation is not dynamics, prediction is not intervention, multimodality is not multiscale) define world-model status. **(paper)**
- **Approach (how they investigated)**: Three motivating experiments plus an L0-L3 capability ladder on three axes. Models and datasets: not available in abstract. **(paper)**
- **Results (what they found)**: FM embeddings carry no future-fate signal, an endpoint predictor loses state-space closure under iteration, and a multimodal model shows no meaningful chromatin-to-RNA intervention effects (qualitative). **(paper)**
- **Implications (what it means)**: Gives falsifiable criteria for a world-model claim, but the ladder needs operational thresholds. **(my take)**

## 2. Problem it solves

"World model" is applied loosely to systems with very different commitments **(paper)**.

## 3. Method

Abstract-only. Three motivating experiments **(paper)**: FM embeddings improve present-state readouts but carry no future-fate signal; a perturbation predictor forecasts endpoints but loses state-space closure under iteration; a multimodal model learns cross-modal association without meaningful chromatin-to-RNA intervention effects. Models/datasets not retrieved.

## 4. Data

Not retrieved.

## 5. Evaluation and results

Qualitative gap demonstrations only, per abstract; numbers not retrieved.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): ladders are ordinal rubrics; risk of arbitrary level thresholds. Needs operational tests.
- What would change my mind: if models at "L2" by the rubric do not outperform L1 models on intervention-held-out tasks, the ladder is not predictive.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md): the PhD's tissue world model must pass these gaps; frames E6.4.
- Hypotheses worth testing:
  - **H-a (E6.4)**: iterating a one-step endpoint predictor k times loses closure (error grows super-linearly) while a latent JEPA-style rollout does not. Null: equal compounding.
  - **H-b**: FM embeddings add no future-fate signal beyond PCA on time-course data (linear probe of clonal fate).

## 8. Related work

Virtual Cells as Causal World Models (NeurIPS 2025), Chreode, CellOS, the Cell perspectives.

## Resources

- [ICML listing](https://icml.cc/virtual/2026/72076) · [Research Square](https://www.researchsquare.com/article/rs-10404367)
