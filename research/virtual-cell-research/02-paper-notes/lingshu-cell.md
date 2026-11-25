# Lingshu-Cell: masked discrete diffusion as a cellular world model

> *Lingshu-Cell: A generative cellular world model for transcriptome modeling toward virtual cells*. arXiv (2026). arXiv: 2603.25240 · Code: not linked on abstract page

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: read (arXiv HTML, first ~93%)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a masked discrete diffusion model over discrete expression tokens for ~18,000 genes, without gene selection, with joint cell-type/donor plus perturbation conditioning **(paper)**.
- **Why it matters for the virtual cell**: a distribution-level generator (not a mean predictor) that handles novel identity-perturbation combinations.
- **Main claim**: reproduces transcriptome distributions, markers and subtype proportions; reports leading results on the Virtual Cell Challenge H1 benchmark and PBMC cytokine responses (self-reported, no metrics in abstract) **(paper)**.

## Research summary

- **Context (already known)**: scFMs are mostly encoders or mean predictors, and autoregressive tokenisation fits sparse non-sequential expression poorly. **(paper)**
- **Gap (not known)**: whether a discrete diffusion generator over ~18,000 genes can reproduce transcriptome distributions and predict unseen identity-perturbation combinations. **(paper)**
- **Hypothesis / what they try to determine**: masked discrete diffusion with joint identity and perturbation conditioning gives faithful distributions and good perturbation prediction. **(paper)**
- **Approach (how they investigated)**: 13-block bidirectional Transformer over 281 expression tokens per gene, with sequence compression and classifier-free guidance; trained on PBMC, 8 tissues, non-human data, VCC H1 and PARSE cytokine data. **(paper)**
- **Results (what they found)**: post-hoc VCC H1 leaderboard: best average rank (8.7 vs 9.1), MAE 0.052, Pearson-delta 0.306, but DES rank 25 of 26; best average score on PBMC cytokines. **(paper)**
- **Implications (what it means)**: strong on some metrics but not DES, and the model is a one-step conditional generator; whether it beats mean/linear baselines is not shown in what was read. **(my take)**

## 2. Problem it solves

Direct control-to-perturbed mappings ignore distributions; sparse non-sequential expression fits poorly with autoregressive tokenization.

## 3. Method

Each cell's UMI counts become 281 tokens per gene (280 bins plus overflow) over ~18,080 genes. Bidirectional LLaMA-style Transformer, 13 blocks, dim 640, 10 heads, RoPE/RMSNorm/SwiGLU; embedding-space compression groups genes (size 8, or 32 for genetic perturbation). Condition tokens (cell type, donor, perturbation) are prepended; classifier-free guidance at sampling (weight 2 for genetic). 256 sampling steps unconditional, 3 conditional **(paper)**.

## 4. Data

Unconditional: 629,701 PBS control PBMCs (PARSE), 2.6M cells from eight CELLxGENE tissues, 247,899 non-human cells. Genetic: VCC H1 training (183,097 cells) plus 323,913 external perturbed cells; test 132,670 cells over 100 held-out targets. Cytokine: PARSE 10M PBMCs, 12 donors, 90 cytokines **(paper)**.

## 5. Evaluation and results

- Unconditional PBMC MMD 0.0088 vs scDiffusion 0.0178 and scVI 0.0343; tissues/species Pearson ~0.997-0.9996 **(paper)**.
- VCC H1 (post-hoc entry): avg rank 8.7 (next 9.1), MAE 0.052 (next 0.086), Pearson-delta 0.306 (first), DES 0.216 (rank 25/26; best 0.356) **(paper)**.
- Cytokine PBMC vs PertMean, STATE, scGPT, scVI: highest average score; exact numbers not given in text read. Ablations: removing CFG hurts; patch 32 scores 0.405 vs 0.292 for patch 8 **(paper)**.

## 6. Critical analysis

- Limitations (paper): population-level metrics cannot assess single-cell plausibility or rare states; faithful distributions do not prove causal mechanism; transcriptomics only; single-gene perturbations only.
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
