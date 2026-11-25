# SCALE: conditional set-level transport at atlas scale

> Chen S, Yu L, ... Gao Z. *SCALE: Scalable Conditional Atlas-Level Endpoint transport for virtual cell perturbation prediction*. arXiv:2603.17380 (2026, v3).

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read (arXiv HTML, first ~100k of 171k characters; remainder not read)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: conditional transport model that treats cells as unordered sets and maps control to treated populations without cell matching: set-aware shared encoder + conditional DiT learning latent transport, with delta-aligned endpoint supervision. **(paper)**
- **Why it matters for the virtual cell**: another set/transport formulation alongside STATE and flow matching; its CRISPR result on strong cell-line effects bears directly on context generalization. **(my take)**
- **Main claim**: recovers expression changes, response direction and population structure across genetic, chemical, developmental and immune perturbations; outperforms competitors on seven metrics on CRISPR data with strong cell-line effects; cytokine prioritization partly confirmed in 3-donor PBMC experiments. **(paper)**

## Research summary

- **Context (already known)**: Perturbation responses are measured on unpaired control and treated cell populations, and set/transport models such as STATE and flow-based models exist. **(paper)**
- **Gap (not known)**: Whether a model can transport whole control populations to treated ones at atlas scale, across genetic, chemical, developmental and immune settings, without cell matching and without collapsing to the mean. **(paper)**
- **Hypothesis / what they try to determine**: A set-aware encoder plus a conditional DiT learning latent delta transport, with endpoint supervision, generalizes across perturbation types. **(paper)**
- **Approach (how they investigated)**: Shared gene-wise attention encoder with DeepSets pooling, conditional DiT on latent delta, autoencoder loss MSE + MMD; benchmarked against STATE, CPA, GEARS and others on Replogle-Nadig, PBMC cytokines, Tahoe-100M, sci-Plex, ZSCAPE and Kang, plus a 3-donor wet-lab test. **(paper)**
- **Results (what they found)**: Led all seven metrics on Replogle (PDCorr 0.7510 vs 0.4132 for STATE); mixed on PBMC and Tahoe (STATE best on some); small gap on Kang (0.932 vs 0.924). No model wins every metric. **(paper)**
- **Implications (what it means)**: Strong on CRISPR cell-line effects, but wins are metric- and dataset-dependent, and the splits predict missing conditions rather than new donors. **(my take)**

## 2. Problem it solves

Control and perturbed cells are unpaired populations. **(paper)**

## 3. Method

Shared set encoder; DiT backbone conditioned on perturbation; endpoint supervision aligned with the delta, no auxiliary delta loss. **(paper)**
- Encoder: shared gene-wise attention per cell, then DeepSets pooling gives a population summary fused back into each cell. Conditions: cell type, perturbation, batch. **(paper)**
- Transport on latent delta ΔZ = Z1 − Z0; autoencoder loss = MSE + λ·MMD. Trained in BioNeMo with LMDB-stored sparse data. **(paper)** Full loss definitions were in the unread tail.

## 4. Data

PBMC cytokine atlas (90 cytokines, 12 donors, 16 types), Tahoe-100M (50 cell lines, >1,100 treatments), Replogle-Nadig CRISPRi (2,024 perturbations, 4 lines), sci-Plex, ZSCAPE (28 perturbations, 5 stages), Kang IFN-β, plus new 3-donor PBMC wet-lab test. **(paper)**

## 5. Evaluation and results

**(paper)** Baselines: STATE, Linear/Latent Additive, Decoder Only, CPA, PRnet, BioLORD, GenePert, SAMS-VAE, scLAMBDA, GEARS. Metrics: MSE, MAE, PDCorr, DEOver, DEPrec, LFCSpear, DirAgr.
- Replogle: best on all seven; PDCorr 0.7510 vs 0.4132 (STATE), DEOver 0.4040 vs 0.2294. Median residual PCC 0.33 vs 0.05; Direction@100 0.88 vs 0.57.
- PBMC: best PDCorr 0.6430, DEOver, DEPrec; STATE best MSE/MAE. Tahoe: SCALE best MSE, MAE, PDCorr 0.8584, DirAgr; STATE best DEOver/DEPrec.
- Kang: ΔPCC 0.932 vs 0.924. ZSCAPE: +10.5 points over STATE across 724 transitions. sci-Plex: lower top-50 MAE than STATE in 94% of 561 conditions.
- Wet lab (5 conditions, 3 donors): IL-12 and IL-15 raised CD69 in CD8 T and NK cells.

## 6. Critical analysis

- Limitations (paper): no model wins every metric; endpoint-only training shows no intermediate biology; developmental accuracy depends on treated-cell count; PBMC split predicts missing conditions for partly measured donors, not new donors; wet lab covers 3 donors and a few cytokines; collapse not ruled out in general.
- Weaknesses I see (my take): baselines are now named but the Kang gap is small and Tahoe DE metrics favour STATE; wet-lab validation is partial and small (3 donors). **(my take)**
- What would change my mind: a win on unseen cell lines after residualization.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: a candidate backbone for E4.3 (distributional) and E4.1 RNA-only arm. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: SCALE beats STATE on held-out cell lines in Replogle-Nadig on E-distance; falsified otherwise. (2) *H*: its advantage depends on cell-line effect size (stratify).

## 8. Related work

STATE ([note](state-arc.md)), CellOT, CellFlow, scDFM ([note](scdfm.md)).

## Resources

- arXiv: https://arxiv.org/abs/2603.17380
