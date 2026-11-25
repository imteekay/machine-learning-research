# CellRFT: reinforcement fine-tuning for single-cell perturbation modeling

> Yan J, Liu L, Guo H, ..., Wang Y. *CellRFT: Reinforcement Fine-Tuning for Single-Cell Perturbation Modeling*. arXiv (2026) · arXiv: 2609.19970 · Code: "will be made available"

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Full text read from arXiv HTML.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read (arXiv HTML v1)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: policy-gradient fine-tuning of pretrained perturbation models using non-differentiable biological evaluations of generated cell populations as rewards, with hierarchical reward aggregation. **(paper)**
- **Why it matters for the virtual cell**: training losses (MSE, MMD) mismatch the biological metrics we care about; RL can optimize the metric directly. **(my take)**
- **Main claim**: works across pretrained models, improves perturbation prediction, and complementary rewards improve criteria beyond those directly optimized; single rewards can trade off against others. **(paper, no numbers seen)**

## Research summary

- **Context (already known)**: Perturbation models are trained with surrogate losses such as MSE and MMD that differ from the biological metrics used to judge them. **(paper)**
- **Gap (not known)**: Whether non-differentiable biological metrics can be used directly as rewards to improve pretrained perturbation models. **(paper)**
- **Hypothesis / what they try to determine**: That reinforcement fine-tuning with hierarchically aggregated rewards improves perturbation prediction across metric families. **(paper)**
- **Approach (how they investigated)**: Flow-GRPO policy gradient on PerturbDiff: 6 candidate sets of 32 cells per perturbation, 11 Cell-Eval metrics in three categories as rewards, averaged within then across categories. Replogle, 2,023 perturbations, HepG2 held out as target. **(paper)**
- **Results (what they found)**: The scratch variant improved all 13 reported metrics (e.g. LFCSpear 0.342 to 0.452, AUPRC 0.210 to 0.270); the finetuned variant improved 9. Single rewards often hurt other metrics, and a four-metric subset showed reward hacking (recall 0.382 to 0.833). **(paper)**
- **Implications (what it means)**: Metric-driven RL helps but is fragile to reward choice; one dataset and no mean-baseline-controlled metric were used. **(my take)**

## 2. Problem it solves

Surrogate losses misalign with biological evaluation criteria. **(paper)**

## 3. Method

Policy-gradient RL based on Flow-GRPO with a clipped objective over diffusion denoising steps. For each perturbation, sample G=6 candidate sets of 32 cells; population-level rewards are normalized within the group and the advantage is shared across cells and denoising steps. Rewards come from Cell-Eval in three categories (global fidelity: PDCorr; DE fidelity: DEOver, DEPrec, DESigRec, DirAgr, LFCSpear, AUROC, AUPRC; perturbation discrimination: PDS L1/L2/cos): 11 of 14 metrics, MSE, MAE and ES excluded. Average within category, then across categories. LR 1e-5. Base models: PerturbDiff (scratch and finetuned). **(paper)**

## 4. Data

Replogle: 2,023 genetic perturbations across K562, RPE1, Jurkat, HepG2; 2,000 HVGs. HepG2 is the target; training uses the other three lines plus 30% of HepG2 perturbations. **(paper)**

## 5. Evaluation and results

**(paper)** Baselines: mean (per cell type, per batch, overall), linear, CPA, STATE, CellFlow, Squidiff.
- Scratch variant improves all 13 metrics: DEOver 0.190 to 0.236, LFCSpear 0.342 to 0.452, AUROC 0.625 to 0.672, AUPRC 0.210 to 0.270.
- Finetuned variant improves 9 metrics and loses on all three perturbation-discrimination metrics.
- Hierarchical reward improves all 11 rewarded metrics (mean gain 6.5 vs 5.9 for a flat mean); single-reward variants usually degrade other metrics.
- A four-metric reward subset showed reward hacking: recall 0.382 to 0.833 while precision and PDCorr fell below baseline.

## 6. Critical analysis

- Limitations (paper, inferred from results; no dedicated section): metric trade-offs, reward-hacking risk, single benchmark (Replogle, HepG2 target), incomplete agreement on which metrics reflect biology; code not yet released.
- Weaknesses I see (my take): reward hacking on metrics that are themselves flawed (e.g. Pearson delta driven by systematic variation); RL on top of a base model adds variance.
- What would change my mind: gains on a held-out metric not in the reward set under strict OOD splits.

## 7. Relevance to my research

- Proposal link: a cheap post-hoc step on the [research proposal](../00-plan/research-proposal.md) H1 model to optimize DE-AUPRC or distributional metrics.
- Hypotheses worth testing:
  - E4.3: RL with a responder-fraction reward improves responder recovery without lowering Pearson delta (falsified if it degrades >2 points).
  - E4.4: reward-optimized models do not improve against a mean baseline when the reward is systematic-variation-controlled.

## 8. Related work

PRiMeFlow ([primeflow-poppert](primeflow-poppert.md)), scDEBART ([scdebart](scdebart.md)), STATE ([state-arc](state-arc.md)).

## Resources

- [arXiv](https://arxiv.org/abs/2609.19970)
