# Perturbation response decomposition: biologically aligned generalization with linear models

> Molina A, Zhang X (AITHYRA). *Perturbation response decomposition enables biologically aligned generalization to unseen perturbations and cellular contexts*. bioRxiv (2026) · DOI: 10.64898/2026.07.24.740459 · Code: not found (Data/Code tab not read)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. First ~100k characters of the full text read.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: splits a pseudobulk response into global shared, perturbation-specific, cell-line-specific and perturbation x cell-line interaction components, then predicts them with Ridge/MLP using DepMap coessentiality priors. **(paper)**
- **Why it matters for the virtual cell**: a direct, quantitative answer to "systematic variation": most of the signal is a template and conserved effect, and simple models with good priors beat complex ones. **(my take)**
- **Main claim**: noise-corrected variance ~27.8% cell-line template, 29.4% conserved perturbation effect, 23.5% interaction, 19.3% noise; Ridge/MLP with DepMap priors match MORPH within cell lines and MLP beats Ridge and MORPH on unseen cell lines; no model recovers the interaction zero-shot; for chemicals, source-cell-line responses (r~0.86) beat Morgan fingerprints (r~0.11). **(paper)**

## Research summary

- **Context (already known)**: Deep models often fail to beat simple baselines on unseen perturbations and contexts. **(paper)**
- **Gap (not known)**: Which components of a perturbation response are predictable, and from what priors, is not known. **(paper)**
- **Hypothesis / what they try to determine**: Whether decomposing responses into template, conserved, specific and interaction parts explains where models succeed and fail. **(paper)**
- **Approach (how they investigated)**: Split-half noise-corrected decomposition of pseudobulk responses on Replogle, Nadig, Norman and Tahoe-100M, then Ridge/MLP with DepMap priors vs scGPT, MORPH, STATE; first ~100k characters read. **(paper)**
- **Results (what they found)**: Variance ~27.8% cell-line template, 29.4% conserved effect, 23.5% interaction, 19.3% noise; simple models match MORPH, none recovers the interaction zero-shot. **(paper)**
- **Implications (what it means)**: Most predictable signal is template and conserved effect, so simple priors suffice there, while cell-line-specific interaction needs target-context data. **(my take)**

## 2. Problem it solves

Deep models often fail to beat simple baselines on unseen perturbations/contexts, and it is unclear which part of the response is predictable. **(paper)**

## 3. Method

Split-half noise-corrected decomposition of (perturbed - non-targeting) pseudobulk; DepMap CRISPRGeneEffect PCA as perturbation descriptor; Ridge/MLP to template-removed responses; Hallmark pathway analysis. **(paper)**

## 4. Data

Replogle K562/RPE1, Nadig HepG2/Jurkat CRISPRi (~97k-189k cells, 1,340-1,537 genes), Norman CRISPRa, Tahoe-100M (5 lines, 268 compounds), DepMap. **(paper)**

## 5. Evaluation and results

Baselines: scGPT, MORPH, STATE, raw DepMap Ridge, additive models (Add+Resid, SALT), Morgan fingerprints, Chemical Checker. Baselines are strong and linear by design. **(paper)** Adding 30% of target-line perturbations recovers much of the interaction. **(paper)**

## 6. Critical analysis

- Limitations (paper): interaction component is not transferable zero-shot.
- Weaknesses I see (my take): pseudobulk evaluation hides single-cell heterogeneity; DepMap priors favor essential genes; four cell lines are mostly cancer lines; authors compare to MORPH/STATE under their own splits.
- What would change my mind: STATE or a foundation model matching Ridge+DepMap on unseen cell lines under identical splits.

## 7. Relevance to my research

- Proposal link: the [research proposal](../00-plan/research-proposal.md) H1 claims ATAC context reduces OOD error; the interaction term (23.5%) is exactly where cell-specific chromatin should help.
- Hypotheses worth testing:
  - E4.2: ATAC features (per cell line) explain a larger share of the interaction component than RNA-only context (falsified if the extra variance explained < 2 points).
  - E4.4: report all models on the template-removed target; a model's gain should survive removal of the global template.

## 8. Related work

Systema (systematic variation), PertAdapt ([pertadapt-closing-the-loop](pertadapt-closing-the-loop.md)), GEARS, STATE ([state-arc](state-arc.md)).

## Resources

- [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.07.24.740459v1)
