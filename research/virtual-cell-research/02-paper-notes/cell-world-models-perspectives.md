# Cell perspectives: world models for biomedicine, and a world model of the virtual cell

> *World models for biomedicine*. Cell 189(19):5845 (2026), doi 10.1016/j.cell.2026.08.032 · *A world model of the virtual cell*. Cell (2026) / ICML 2026 listing. Code: n/a

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed (Cell pages returned 403; only secondary summaries and the ICML abstract of the second paper)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: two position pieces. The first argues biomedical AI should move beyond classification and single-step prediction to models that represent states, accept interventions and simulate future states (secondary summary). The second defines a virtual cell world model as a generative system that simulates cell outcomes under interventions, contrasting it with task-specific expression predictors (ICML abstract).
- **Why it matters for the virtual cell**: it is the field's vocabulary for the "next step" the PhD targets.
- **Main claim**: abstract-only; no results (perspectives).

## 2. Problem it solves

Fragmented use of "world model" and "virtual cell" **(paper, secondary)**.

## 3. Method

Perspectives; the second paper proposes an operational definition built on world-model architecture **(paper, abstract-only)**. No further detail.

## 4. Data

None.

## 5. Evaluation and results

None retrieved. Not empirical.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): definitions without benchmarks can be unfalsifiable; read together with the three-gaps paper for testable criteria.
- What would change my mind: a quantitative benchmark showing models meeting the definition outperform predictors on held-out interventions.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md): framing and citations for the tissue world model.
- Hypotheses worth testing:
  - **H-a (E6.4)**: models satisfying the "simulate under intervention" definition show lower multi-step error than one-step predictors; else the definition adds no predictive content.

## 8. Related work

[vcwm-three-gaps](vcwm-three-gaps.md), Towards World Models in Biomedical Research (arXiv 2606.05925).

## Resources

- [Cell, World models for biomedicine](https://www.cell.com/cell/fulltext/S0092-8674(26)01005-6) · [Cell, A world model of the virtual cell](https://www.cell.com/cell/fulltext/S0092-8674(26)01015-9) · [ICML listing](https://icml.cc/virtual/2026/80215)
