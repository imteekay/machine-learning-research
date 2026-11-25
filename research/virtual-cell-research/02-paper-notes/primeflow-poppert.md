# PRiMeFlow and PopPert: flow matching and population-level distributions for heterogeneous perturbation responses

> PRiMeFlow: Yan Z, Wu Y, et al., Stark R. *PRiMeFlow: Capturing Complex Expression Heterogeneity in Perturbation Response Modelling*. arXiv (2026) · arXiv: 2604.13986 · Code: not found. PopPert: Wang H, Qi J, Feng H, Lai B. *PopPert: Population-level Joint-Distribution Modeling for Single-Cell Perturbation Prediction*. arXiv (2026) · arXiv: 2609.01357 · Code: [whd1125/PopPert](https://github.com/whd1125/PopPert)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Both notes updated from arXiv HTML full text; appendices partly unread.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: read (arXiv HTML full text; PRiMeFlow last ~39k characters and PopPert last ~1k not read)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: PRiMeFlow is a flow-matching model in gene-expression space (U-Net velocity field) for genetic and small-molecule perturbations, scaled to a multi-dataset atlas with pretrain/finetune. PopPert instead predicts shifts in the parameters of a population-level joint distribution (low-rank Gaussian copula) from a control population. **(paper)**
- **Why it matters for the virtual cell**: two opposite answers to "what is the prediction target": a per-cell generative map vs a population distribution. This is the heterogeneity question behind E4.3. **(my take)**
- **Main claim**: PRiMeFlow reports strong results on PerturBench and on H1 hESC from the ARC Virtual Cell Challenge; PopPert reports superior DE recovery, effect estimation and distribution matching, with lower sensitivity to single-cell noise. **(paper, abstract-level, no numbers seen)**

## Research summary

- **Context (already known)**: Perturbation models (CPA, SAMS-VAE, STATE, scDFM) often predict mean shifts or assume cell-to-cell correspondence that scRNA-seq cannot provide. **(paper)**
- **Gap (not known)**: Whether per-cell flow matching or population-level joint distributions better capture heterogeneous responses is not settled. **(paper)**
- **Hypothesis / what they try to determine**: PRiMeFlow: flow matching directly in expression space captures heterogeneity. PopPert: predicting shifts of population distribution parameters avoids pairing. **(paper)**
- **Approach (how they investigated)**: PRiMeFlow: U-Net flow matching with CFG on PerturBench, Norman19 and the ARC VCC H1 data with an atlas pretraining. PopPert: zero-inflated mixture marginals plus rank-32 Gaussian copula predicted by a Transformer; Adamson, Norman, Replogle, sci-Plex3. **(paper)**
- **Results (what they found)**: PRiMeFlow gets large MMD and DEG-recall gains but is weaker than CPA/SAMS-VAE on pseudobulk metrics and lost to simple baselines on some Norman metrics; PopPert best in 18 of 24 genetic metric combinations, mixed on others. **(paper)**
- **Implications (what it means)**: Distributional models win on distribution metrics but not mean metrics, so metric choice decides the ranking; no multi-seed variance was reported for PopPert. **(my take)**

## 2. Problem it solves

Perturbed cells are never paired with their own control (scRNA-seq is destructive), and responses are heterogeneous; mean-shift models discard that. PopPert argues cell-to-cell correspondence assumptions are a mismatch. **(paper)**

## 3. Method

PRiMeFlow: flow matching directly on expression, U-Net velocity, multi-dataset pretraining then finetuning, ablations on design choices. PopPert: parameterizes joint gene distribution with a low-rank Gaussian copula; maps (control population, condition) to parameter shifts; can sample synthetic perturbed cells. PRiMeFlow: log1p expression, Gaussian source, U-Net velocity with gene-identity embeddings, summed one-hot perturbation encoding, 100 Euler steps, classifier-free guidance. PopPert: zero-inflated K=2 truncated Gaussian mixture marginals, rank-32 Gaussian copula, 4-layer Transformer decoder predicting control-referenced residuals, 512 cells sampled per pair.

## 4. Data

PRiMeFlow: Srivatsan20 (178k cells), Jiang24 (1.6M), Norman19 (91k), ARC VCC H1 hESC (300 perturbations); atlas of ~7.0M cells from eight datasets. PopPert: Adamson, Norman, Replogle (HepG2 held out), sci-Plex3 (9 held-out drugs).

## 5. Evaluation and results

PRiMeFlow: Srivatsan20 MMD 0.13 vs CPA 2.4 and Linear 2.2; DEG recall 0.26 vs 0.0073 (CPA); Norman19 DEG recall 0.61 vs 0.018 (Linear); weaker than CPA/SAMS-VAE on pseudobulk cosine/RMSE; VCC private test DES 0.228, PDS 0.747, MAE 0.086 for the 250-pt variant (lowest average rank on the Generalist board per authors). PopPert: best in 18 of 24 genetic metric combinations; Replogle Ovl@200 0.373 vs 0.183, MSE 36% lower; sci-Plex3 MSE 22.5% lower; loses on some distributional metrics to STATE/scDFM. Baselines include Context-Mean and Perturb-Mean for PopPert, Linear Additive for PRiMeFlow. **(paper)**

## 6. Critical analysis

- Limitations (paper): PRiMeFlow: U-Net is theoretically ill-suited to unordered genes, simpler baselines win on some Norman19 metrics, CFG trades DEG recall against distribution fit, manual hyperparameters, VCC 250-pt model finetunes on test perturbations (no labels). PopPert: fixed shared loadings, Gaussian copula only, no calibrated uncertainty, O(G^2) preprocessing, single-dataset ablations without variance.
- Weaknesses I see (my take): a Gaussian copula may miss multimodal responder/non-responder structure; flow matching in 2000+ dim expression space is costly and metric-sensitive; neither is shown (from abstracts) to beat a linear baseline OOD.
- What would change my mind: PopPert beating a flow model on a bimodal-response dataset, or PRiMeFlow's atlas pretraining helping on a held-out cell line vs training from scratch.

## 7. Relevance to my research

- Proposal link: the [research proposal](../00-plan/research-proposal.md) H1 needs a distribution-aware target; these are two candidate heads on top of an RNA+ATAC encoder (E4.1, E4.3).
- Hypotheses worth testing:
  - E4.3: on a dataset with a known responder/non-responder split, a flow model recovers the responder fraction within 5 points while a copula population model does not (falsified if the copula matches).
  - Population-level parameter prediction is more robust than per-cell prediction when cells per condition < 50 (falsified if the gap does not grow as cells are subsampled).

## 8. Related work

CellFlow, STATE ([state-arc](state-arc.md)), scGen ([scgen](scgen.md)), PerturBench.

## Resources

- [PRiMeFlow arXiv](https://arxiv.org/abs/2604.13986) · [PopPert arXiv](https://arxiv.org/abs/2609.01357) · [PopPert code](https://github.com/whd1125/PopPert)
