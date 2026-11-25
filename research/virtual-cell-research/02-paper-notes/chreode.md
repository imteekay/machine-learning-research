# Chreode: a Waddington-structured one-step cell world model

> *Chreode: A Cell World Model for One-Step Temporal Dynamics and Perturbation Prediction*. arXiv (2026). arXiv: 2605.28111 · Code: not linked on abstract page

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed (abstract-only)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a structured residual transition operator decomposed into downhill landscape flow, rotational dynamics and stochastic spread; shared scVI encoder with a DiT dynamics backbone; distributional evolution moved to training time for one-pass generation **(paper)**.
- **Why it matters for the virtual cell**: dynamics learned on a large atlas transfer to perturbation tasks.
- **Main claim**: pretrained on 2.4M-cell mouse embryo atlas (7 datasets); beats scratch, PI-SDE and PRESCIENT in Sinkhorn on Weinreb and Veres; as an embedding for GEARS lowers Norman DE20 MSE 0.2121 to 0.1858 (12.4%); zero-shot clonal fate scores competitive with dynamic OT **(paper)**.

## 2. Problem it solves

Dynamic models are slow at inference and trained per dataset; perturbation predictors ignore developmental dynamics.

## 3. Method

As above; no further detail retrieved.

## 4. Data

2.4M mouse embryonic atlas pretraining; Weinreb hematopoiesis, Veres islets, Norman Perturb-seq for evaluation **(paper)**.

## 5. Evaluation and results

Numbers above. The GEARS gain is relative to GEARS only, not to a mean or linear baseline **(my take)**; shared-vocabulary DE20 subset only.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): one-step generation does not test multi-step stability; the GEARS comparison does not show gains over simple baselines on Norman.
- What would change my mind: a linear baseline matching 0.1858 DE20 MSE.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md): evidence that dynamics pretraining helps perturbation prediction; named in E6.4.
- Hypotheses worth testing:
  - **H-a (E6.4)**: Chreode applied recursively has higher horizon-k error than a latent-space rollout (CellOS-style). Null: equal.
  - **H-b (E4.x)**: Chreode embeddings in GEARS beat a PCA-embedding GEARS on Norman; else the gain is just a better embedding.

## 8. Related work

PRESCIENT, PI-SDE, GEARS, scVI, AlphaCell.

## Resources

- [arXiv](https://arxiv.org/abs/2605.28111) · [HF page](https://huggingface.co/papers/2605.28111.md)
