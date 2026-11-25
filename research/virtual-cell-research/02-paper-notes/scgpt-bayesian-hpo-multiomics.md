# Bayesian HPO for scGPT multi-omics fine-tuning: tuning matters more than defaults suggest

> Tay DYJ, Le NQK, Chua MCH. *Bayesian hyperparameter optimization improves scGPT fine-tuning for single-cell multi-omics integration*. Bioinformatics 42(7), btag374 (2026). DOI: [10.1093/bioinformatics/btag374](https://doi.org/10.1093/bioinformatics/btag374) · Code: [daren642/scGPT_multiomic_tuning](https://github.com/daren642/scGPT_multiomic_tuning)
>
> Open access (CC BY 4.0); PDF not committed.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context), [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: read (full text via publisher page, summarised)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: TPE (Hyperopt) search over scGPT fine-tuning hyperparameters (layers 2-7, LR 1e-5 to 1e-3, epochs 20/30/40, batch 16/32) for RNA+protein integration; 30 trials (10 random + 20 TPE). **(paper)**
- **Why it matters for the virtual cell**: scGPT is the proposal's backbone; if defaults are badly off, any "scGPT vs baseline" comparison (E2.1, E2.2, E4.1) is confounded by tuning budget. **(my take)**
- **Main claim**: tuned scGPT beats default scGPT and matches/exceeds Seurat v4 on integration. **(paper)**

## Research summary

- **Context (already known)**: scGPT is fine-tuned with hand-picked defaults for multi-omic integration, and tuning effects are rarely quantified. **(paper)**
- **Gap (not known)**: Whether systematic hyperparameter search materially improves scGPT RNA+protein integration, and whether it matches Seurat v4. **(paper)**
- **Hypothesis / what they try to determine**: TPE-based Bayesian optimization of layers, learning rate, epochs and batch size gives better integration than default scGPT. **(paper)**
- **Approach (how they investigated)**: 30-trial Hyperopt search (10 random, 20 TPE) on validation loss, evaluated on BMMC CITE-seq and GSE194122 with AvgBIO, PCR and ARI. **(paper)**
- **Results (what they found)**: Tuned scGPT raised BMMC AvgBIO 0.59 to 0.67 and GSE194122 AvgBIO 0.19 to 0.60, and slightly exceeded Seurat v4 (0.60) on BMMC. **(paper)**
- **Implications (what it means)**: Tuning budget can confound model comparisons; however the near-failed default on GSE194122 and missing PCA/scVI/totalVI baselines mean the size of the effect is uncertain. **(my take)**

## 2. Problem it solves

Manual, irreproducible fine-tuning of scGPT for multi-omic integration. **(paper)**

## 3. Method

- scGPT_human checkpoint; fixed hidden 512, 8 heads, dropout 0.2, StepLR γ=0.95; GEP, GEPC and DAR losses on, CLS off to avoid label leakage during tuning. **(paper)**
- Objective: validation loss; best reported config 7 layers, LR 3.21e-4, 40 epochs, batch 32. **(paper)**
- Consumer GPU (RTX 3060 Ti). **(paper)**

## 4. Data

BMMC CITE-seq (NeurIPS 2021 multimodal benchmark, multiple donors) and GSE194122 (12 donors; RNA, ATAC, protein). **(paper)**

## 5. Evaluation and results

| Dataset | Metric | Default | Tuned |
|---|---|---|---|
| BMMC | AvgBIO | 0.59 | 0.67 |
| BMMC | PCR (batch) | 0.33 | 0.52 |
| BMMC | ARI | 0.48 | 0.66 |
| GSE194122 | AvgBIO | 0.19 | 0.60 |
| GSE194122 | ARI | 0.007 | 0.63 |

Seurat v4 BMMC AvgBIO 0.60. Wilcoxon + FDR significant for ARI and AvgBIO on BMMC. **(paper)** Baselines are default scGPT and Seurat only; no PCA/scVI/totalVI baseline. **(my take)**

## 6. Critical analysis

- Limitations (paper): limited hardware, shallow search space, AvgBIO misses rare cells/trajectories, label noise, single-dataset generalisation.
- Weaknesses I see (my take): internal inconsistencies (3 vs 5 seeds, default LR 1e-3 vs 0.003, default epochs 25 outside the search grid, limitations section says one dataset while two are reported). A GSE194122 default ARI of 0.007 looks like a failed run, not a fair default. Tuning on validation loss and reporting on the same data risks optimistic estimates.
- What would change my mind: the tuned config transferring to a *third* dataset without retuning, with gains over a scVI/totalVI baseline given the same tuning budget.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: methodological warning for the Master's: compare RNA-only vs RNA+ATAC (E4.1) under *equal* tuning budgets, otherwise the "context" effect could be a tuning effect. Also relevant to E2.2 (does fine-tuning close the gap). **(my take)**
- Hypotheses worth testing:
  - **H-hpo1** (E2.1/E2.2): the gap between scGPT and PCA/scVI embeddings shrinks by >50% when both receive the same HPO budget. *Null*: the gap is unchanged by equal tuning.
  - **H-hpo2** (E4.1 control): the ATAC-vs-RNA-only difference keeps its sign across the top-5 HPO configs of each arm. *Null*: the sign flips, i.e. the effect is tuning-sensitive.

## 8. Related work

[scGPT](scgpt.md), scEval, BBOmix, Seurat v4 WNN, totalVI.

## Resources

- Paper: https://academic.oup.com/bioinformatics/article/42/7/btag374/8707330
- Code: https://github.com/daren642/scGPT_multiomic_tuning
