# SCALE: conditional set-level transport at atlas scale

> Chen S, Yu L, ... Gao Z. *SCALE: Scalable Conditional Atlas-Level Endpoint transport for virtual cell perturbation prediction*. arXiv:2603.17380 (2026, v3).

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. This note is **abstract-only**: the full text was not retrieved (paywall / rate limit), so no claim below comes from the methods or results sections.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: conditional transport model that treats cells as unordered sets and maps control to treated populations without cell matching: set-aware shared encoder + conditional DiT learning latent transport, with delta-aligned endpoint supervision. **(paper)**
- **Why it matters for the virtual cell**: another set/transport formulation alongside STATE and flow matching; its CRISPR result on strong cell-line effects bears directly on context generalization. **(my take)**
- **Main claim**: recovers expression changes, response direction and population structure across genetic, chemical, developmental and immune perturbations; outperforms competitors on seven metrics on CRISPR data with strong cell-line effects; cytokine prioritization partly confirmed in 3-donor PBMC experiments. **(paper)**

## 2. Problem it solves

Control and perturbed cells are unpaired populations. **(paper)**

## 3. Method

Shared set encoder; DiT backbone conditioned on perturbation; endpoint supervision aligned with the delta, no auxiliary delta loss. **(paper)**

## 4. Data

genetic, chemical, developmental, immune datasets (names abstract-only); new 3-donor PBMC cytokine validation. **(paper)**

## 5. Evaluation and results

seven metrics, unnamed in abstract; no numbers recorded (abstract-only).

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): 'competing methods' unnamed; wet-lab validation is partial and small (3 donors). **(my take)**
- What would change my mind: a win on unseen cell lines after residualization.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: a candidate backbone for E4.3 (distributional) and E4.1 RNA-only arm. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: SCALE beats STATE on held-out cell lines in Replogle-Nadig on E-distance; falsified otherwise. (2) *H*: its advantage depends on cell-line effect size (stratify).

## 8. Related work

STATE ([note](state-arc.md)), CellOT, CellFlow, scDFM ([note](scdfm.md)).

## Resources

- arXiv: https://arxiv.org/abs/2603.17380
