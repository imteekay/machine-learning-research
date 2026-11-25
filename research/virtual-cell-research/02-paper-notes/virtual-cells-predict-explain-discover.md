# Virtual Cells: Predict, Explain, Discover: a drug-discovery framework for what a virtual cell must do

> Noutahi E, Hartford J, ..., Earnshaw B (Valence Labs / Recursion). *Virtual Cells: Predict, Explain, Discover*. arXiv (2025). [PDF](../papers/virtual-cells-predict-explain-discover.pdf) · arXiv: [2505.14613](https://arxiv.org/abs/2505.14613) · Code: none (perspective) · [Video](https://www.youtube.com/watch?v=QSfVjsa1i4g)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 0](../00-plan/roadmap.md#phase-0-foundations), [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a perspective arguing that a useful virtual cell must (1) **predict** functional responses to perturbations, (2) **explain** them as changes in biomolecular interactions, and (3) support **discovery** through lab-in-the-loop falsification. **(paper)**
- **Why it matters for the virtual cell**: it gives this roadmap its three goals and gives concrete design rules (predict *relative* change, use negative controls) that directly shape how E4.x targets are defined. **(my take)**
- **Main claim**: prediction alone is not enough for drug discovery. Models must give falsifiable mechanistic explanations and be improved by experiments chosen to falsify them. **(paper)**

## Research summary

- **Context (already known)**: Roughly 9 in 10 drugs entering trials fail, atomistic cell simulation is intractable, and data-driven virtual cells lack an agreed goal. **(paper)**
- **Gap (not known)**: What a virtual cell must do to be useful for drug discovery. **(paper)**
- **Hypothesis / what they try to determine**: A virtual cell must predict, explain and support discovery via lab-in-the-loop falsification. **(paper)**
- **Approach (how they investigated)**: Perspective with design principles (relative change, negative controls, Popperian active learning) and L1-L3 capability levels. **(paper)**
- **Results (what they found)**: No experiments; benchmark recommendations and principles only. **(paper)**
- **Implications (what it means)**: Gives concrete target definitions for E4.x, but explanation lacks a scoring rule. **(my take)**

## 2. Problem it solves

About 9 in 10 drugs that enter clinical trials fail. Full atomistic simulation of a cell (~100 trillion atoms) is intractable, and even the minimal cell JCVI-syn3A could not be simulated end to end with MD. Data-driven virtual cells are the alternative, but the field lacks agreement on what they should do. **(paper)**

## 3. Method

A framework, not a model. **(paper)**
- **Predict**: estimate outcomes under intervention (causal sense). Readouts: expression, morphology, protein activity.
- **Explain**: structured, testable accounts in terms of biomolecular interactions (binding, PTMs, transcriptional regulation, signaling). Not post-hoc saliency. "Explanatory pluralism": mechanistic, semantic or statistical-structural explanations are fine if falsifiable.
- **Discover**: follows from predict + explain via lab-in-the-loop.
- **Design principles**:
  1. Predict *relative* change from the pre-perturbation state; use negative controls to remove the shared (housekeeping / stress) response. This also absorbs batch effects.
  2. Explain via dynamic changes to interactions, using structure tools (AlphaFold, Boltz) and targeted simulation.
  3. Active learning framed as Popperian falsification: choose experiments that could falsify the model; update on failure.
- **Capability levels**: L1 = specific biology at one time point; L2 = core biology across environments and time; L3 = unobserved biology in spatially organized systems with causal interpretability.

## 4. Data

No dataset introduced. Mentions Recursion's automated phenomics lab (~2.2M samples/week of microscopy). **(paper)**

## 5. Evaluation and results

No experiments. Benchmark recommendations **(paper)**: cover functional responses (multi-modal), cell contexts (intrinsic and extrinsic), perturbations with different amounts of known mechanism, and all three P-E-D capabilities. Watch for leakage from literature-derived gene embeddings. Appendix A lists complementary local/global metrics, biologically informed splits, prospective evaluation (I did not read the appendix body in detail). Cites GEARS, CPA, scLAMBDA, CellFlow, TxPert as current predictors without evaluating them.

## 6. Critical analysis

- Limitations (paper): explanations and novelty are much harder to benchmark than predictions; scientist-AI agents and virtual patients are "only a vision".
- Weaknesses I see (my take):
  - Industry perspective: no numbers to show that any current model reaches Level 1.
  - "Explain" is hard to operationalize. Without a scoring rule, any model can claim explanations.
  - The "relative change + negative controls" rule is the most actionable point, and it matches the Systema critique (shared response inflates scores).
- What would change my mind: a benchmark where models that give mechanistic explanations do *not* predict better on unseen perturbations would weaken the claim that explanation and prediction should be built together.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses:
  - The Master's hypothesis is an "explain" claim as much as a "predict" claim: ATAC should help *because* it encodes accessible regulatory DNA. E4.2 (where the context helps) is the explanation test.
  - The PhD hypothesis targets Level 3 (spatially organized systems).
  - Principle 1 sets how to define targets in E4.1 and E4.4: predict delta vs matched controls, and report scores after removing the average perturbation response.
- Hypotheses worth testing:
  - **H-PED-1 (relative targets)**: training on delta-from-control vs on absolute post-perturbation expression gives lower error on unseen knockouts after Systema-style correction. *Null*: no difference. Cheap ablation inside E4.1.
  - **H-PED-2 (explanation as a test)**: if ATAC helps (E4.1), the gain should be largest for targets of TFs with peaks linked to their DE genes (E4.2). If the gain is uniform across genes, the "regulatory" explanation is not supported.
  - **H-PED-3 (falsification-driven selection)**: in a simulated loop on Replogle 2022, picking knockouts with highest ensemble disagreement reaches a given error with fewer "experiments" than random choice (E6.3).

## 8. Related work

- [AIVC manifesto](ai-virtual-cell-priorities.md): broader academic vision.
- [Interpretation, extrapolation and perturbation of single cells](interpretation-extrapolation-perturbation-review.md): methods taxonomy for the predict and explain parts.
- [Glyph](glyph.md): one route to the "explain" capability.

## Resources

- arXiv: https://arxiv.org/abs/2505.14613
- Video: https://www.youtube.com/watch?v=QSfVjsa1i4g
