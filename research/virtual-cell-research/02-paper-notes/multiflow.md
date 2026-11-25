# MultiFlow: coupled RNA+ATAC flow matching for unseen cellular contexts

> Liu Q lab. *MultiFlow: coupled flow matching for predicting single-cell multiomic perturbation responses in unseen cellular contexts*. bioRxiv (2026). DOI: 10.64898/2026.08.20.746112 · Code: [liuq-lab/MultiFlow](https://github.com/liuq-lab/MultiFlow)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. This note is **abstract-only**: the full text was not retrieved (paywall / rate limit), so no claim below comes from the methods or results sections.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: coupled flow matching that jointly models paired RNA and chromatin accessibility, conditioned on perturbation and control-derived cell-state representations, to predict multiomic responses in contexts unseen in training. **(paper)**
- **Why it matters for the virtual cell**: the closest prior work to the Master's thesis: multi-omic (ATAC) + unseen contexts. The thesis must be positioned against it. **(my take)**
- **Main claim**: beats modality-specific methods on both expression and accessibility prediction, while preserving peak–gene and cross-modal neighborhood structure. **(paper)**

## 2. Problem it solves

Perturbation models predict one modality and fail on new contexts; chromatin and expression responses are coupled. **(paper)**

## 3. Method

Two coupled flows (RNA, ATAC) sharing conditioning on perturbation + control-state embedding. Coupling mechanism abstract-only. **(paper)**

## 4. Data

paired multiome perturbation data (names abstract-only; likely Perturb-multiome-type screens). **(uncertain)**

## 5. Evaluation and results

abstract-only: no numbers; baselines are modality-specific perturbation methods. Unknown whether an RNA-only ablation of the same model was run — the key test for H1.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): it *predicts* ATAC as an output; the thesis uses ATAC as *input context*. Whether ATAC conditioning lowers RNA error is a different, still-open question unless the paper ablates it. **(my take)**
- What would change my mind: if MultiFlow already shows RNA+ATAC-conditioned < RNA-only on unseen contexts with linear baselines, H1 is pre-empted and the thesis should pivot to mechanism (E4.2) or imputed ATAC.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: directly on H1 and E4.1; read full text first and extract its splits. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: an RNA-only MultiFlow ablation is significantly worse on unseen contexts (E4.1 design); falsified if equal. (2) *H*: MultiFlow's gain concentrates in TF/chromatin-regulator knockouts (E4.2). (3) *H*: with imputed ATAC (E3.2) it retains ≥50% of the gain.

## 8. Related work

scDFM ([note](scdfm.md)), CellFlow, CFM-GP, STATE ([note](state-arc.md)).

## Resources

- Sciety: https://sciety.org/articles/activity/10.64898/2026.08.20.746112
- Repo: https://github.com/liuq-lab/MultiFlow
