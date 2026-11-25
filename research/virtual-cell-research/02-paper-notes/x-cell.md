# X-Cell: scaled diffusion language model for causal perturbation prediction

> Xaira Therapeutics. *X-Cell: Scaling Causal Perturbation Prediction Across Diverse Cellular Contexts via Diffusion Language Models*. bioRxiv (2026). DOI: 10.64898/2026.03.18.712807 · Project: [xaira-therapeutics.github.io/X-Cell](https://xaira-therapeutics.github.io/X-Cell) · 55M 'Mini' weights on Hugging Face

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Preprint full text not retrieved; this note adds the project page (re-read) to the abstract.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed (partial: project page only)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: diffusion language model over sets of cells, trained on interventional (CRISPRi) data with distribution-matching objectives; largest version X-Cell-Ultra has 4.9B parameters. **(paper / project page)**
- **Why it matters for the virtual cell**: the first evidence of LLM-like power-law scaling for causal perturbation prediction, and a new genome-wide multi-context dataset (X-Atlas/Pisces). **(my take)**
- **Main claim**: loss follows a power law (α ≈ 0.32, R² 0.96 over 83M–3.1B params); Pearson Δ 0.51 on held-out iPSC perturbations; zero-shot prediction of T-cell inactivating perturbations confirmed in an independent primary T-cell screen. **(project page)**

## Research summary

- **Context (already known)**: Perturbation models trained on one context do not transfer, and scaling had not been tested systematically. **(paper)**
- **Gap (not known)**: Whether LLM-like scaling holds for causal perturbation prediction across contexts. **(paper)**
- **Hypothesis / what they try to determine**: Scaling a diffusion language model on large interventional data improves held-out perturbation prediction. **(paper)**
- **Approach (how they investigated)**: Diffusion LM over cell sets trained on X-Atlas/Pisces (25.6M CRISPRi cells). Project page: cross-attention integrates multimodal biological priors; X-Cell Mini (55M) is initialised from scGPT; scaling runs span 83M to 4.9B parameters; test-time adaptation on unlabeled controls is used for new cell types. **(project page)**
- **Results (what they found)**: Power-law loss scaling (alpha about 0.32, R2 0.96), Pearson delta 0.51 on held-out iPSC, T-cell zero-shot validation (project page). **(paper)**
- **Implications (what it means)**: Scale is a strong arm of the context-vs-scale question, but loss scaling may not transfer to residualized metrics. **(my take)**

## 2. Problem it solves

Perturbation models trained on one context do not transfer; data and model scale had not been tested systematically. **(paper)**

## 3. Method

Masked/discrete diffusion LM over gene-expression tokens of cell sets, conditioned on perturbation and context; distribution-matching loss. Abstract-level only. **(paper)**

## 4. Data

X-Atlas/Pisces: 25.6M perturbed single-cell transcriptomes, genome-wide CRISPRi, seven CRISPRi screens per the project page: HCT116 (18,924 perturbations), HEK293T (18,312), HepG2 (9,735), iPSC (10,095), Jurkat resting (10,872), Jurkat active (10,878), iPSC multi-differentiation (12,175). Subset released. **(project page)**

## 5. Evaluation and results

Pearson Δ 0.51 (iPSC held-out); scaling curves; T-cell validation. The project page names only an unnamed 'next-best method' (the 0.51 is said to be over 5x it), reports no error bars or split definitions, and lists LRBA and APPL2 as predicted T-cell regulators confirmed by an independent primary screen. **(project page)**

## 6. Critical analysis

- Limitations (paper): not retrieved from the paper. The project page states weights are 'coming soon' for anything beyond Mini, the preprint link is a placeholder, and the licence is CC BY-NC-SA 4.0. **(project page)**
- Weaknesses I see (my take): loss scaling is not the same as task scaling; Pearson Δ is inflated by generic responses. Industrial dataset, partial release, so results are hard to reproduce. **(my take)**
- What would change my mind: scaling curves on a residualized metric (E4.4) that still follow a power law.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: if scale alone closes the OOD-context gap, H1's argument for explicit epigenetic context weakens; X-Cell is the 'scale' arm of 'context vs scale'. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: X-Cell-Mini + ATAC context beats X-Cell-Mini alone on held-out contexts; falsified if no gain. (2) *H*: residual Pearson (Systema) saturates at much smaller model size than raw loss.

## 8. Related work

STATE ([note](state-arc.md)), LPM ([note](large-perturbation-model-lpm.md)), scGPT ([note](scgpt.md)).

## Resources

- Project: https://xaira-therapeutics.github.io/X-Cell
