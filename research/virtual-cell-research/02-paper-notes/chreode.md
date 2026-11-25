# Chreode: a Waddington-structured one-step cell world model

> *Chreode: A Cell World Model for One-Step Temporal Dynamics and Perturbation Prediction*. arXiv (2026). arXiv: 2605.28111 · Code: not linked on abstract page

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: read (arXiv HTML v1; last ~38k characters of the appendix not read)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a structured residual transition operator decomposed into downhill landscape flow, rotational dynamics and stochastic spread; shared scVI encoder with a DiT dynamics backbone; distributional evolution moved to training time for one-pass generation **(paper)**.
- **Why it matters for the virtual cell**: dynamics learned on a large atlas transfer to perturbation tasks.
- **Main claim**: pretrained on 2.4M-cell mouse embryo atlas (7 datasets); beats scratch, PI-SDE and PRESCIENT in Sinkhorn on Weinreb and Veres; as an embedding for GEARS lowers Norman DE20 MSE 0.2121 to 0.1858 (12.4%); zero-shot clonal fate scores competitive with dynamic OT **(paper)**.

## Research summary

- **Context (already known)**: Dynamic cell models are slow at inference and trained per dataset, and perturbation predictors ignore developmental dynamics. **(paper)**
- **Gap (not known)**: Whether dynamics pretrained on a large atlas with a structured transition operator transfers to time-course and perturbation tasks. **(paper)**
- **Hypothesis / what they try to determine**: That a one-step operator split into landscape flow, rotation and noise gives better, faster dynamics than per-dataset SDE models. **(paper)**
- **Approach (how they investigated)**: Pretrain a scVI encoder plus DiT on 2.48M mouse embryo cells; fine-tune on Weinreb and Veres; replace the GEARS embedding on Norman; ablate components. **(paper)**
- **Results (what they found)**: Better Sinkhorn W2 than scratch, PI-SDE and PRESCIENT (e.g. Weinreb d4 1.5133 vs 1.7452 PI-SDE), clonal-fate r 0.468 zero-shot, Norman DE20 MSE 0.2121 to 0.1858, and about 3x faster inference than PRESCIENT. **(paper)**
- **Implications (what it means)**: Dynamics pretraining is a useful embedding, but no mean or linear baseline was tested on Norman and no multi-step rollouts were evaluated. **(my take)**

## 2. Problem it solves

Dynamic models are slow at inference and trained per dataset; perturbation predictors ignore developmental dynamics.

## 3. Method

Residual update z + a(D)(-grad U + Sz + sigma*eps), with a learned potential, an antisymmetric rotation term S = PQ^T - QP^T, learned noise and a time gate a(D) = 1 - exp(-D/tau). Shared DiT backbone (hidden 384, depth 12). Loss: MMD + Sinkhorn W2 + drifting-field term + downhill regularizer, trained on unpaired snapshots over all ordered timepoint pairs. Two stages: scVI encoder (d=128), then frozen-encoder DiT (3,356 steps). **(paper)**

## 4. Data

2,477,217 mouse embryonic cells from seven datasets (0-19 dpf, 88 timepoints), 16,520 one-to-one orthologs. Evaluation: Weinreb, Veres, Norman via GEARS, and CellStream/MOSTA for a velocity check. **(paper)**

## 5. Evaluation and results

Numbers above plus **(paper)**: Weinreb W2 d4 1.5133 vs scratch 1.6387, PI-SDE 1.7452, PRESCIENT 1.9096; d6 1.6884 vs 2.0478, 1.8401, 1.8846. Veres avg 2.6171 vs 2.8033, 2.8296, 3.3219 (t7 within seed noise of PI-SDE). Clonal fate r 0.4680 vs scDiffEq 0.4627, moscot 0.4624, static-DiT 0.3825. Norman: static-DiT replacement 0.1936, scVI 0.2126. Inference 64.79 ms vs PRESCIENT 194 ms and CellFlow 410 ms. Ablations: single-delta training and removing drift or downhill losses worsen Veres (3.79, 3.63, 3.68). The GEARS gain is relative to GEARS only, not to a mean or linear baseline **(my take)**; shared-vocabulary DE20 subset only.

## 6. Critical analysis

- Limitations (paper): mouse embryo only; time results use fine-tuning, only clonal fate is zero-shot; perturbation reuses GEARS with no standalone operator; about 10^7 parameters and no comparison with AlphaCell or X-Cell; no Geneformer or scGPT benchmark; latent-space evaluation with no uncertainty calibration; weights not released; single pretraining seed.
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
