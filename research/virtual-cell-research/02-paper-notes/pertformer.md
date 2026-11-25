# PertFormer: a 3B multimodal foundation model for zero-shot perturbation prediction

> Yang Z, Fan X, Lan M, Tang X, Zheng Z, Liu B, You Y, Tian L, Church G, Liu X, Gu F. *Multimodal foundation model predicts zero-shot functional perturbations and cell fate dynamics*. bioRxiv (2024). DOI: [10.1101/2024.12.19.629561](https://www.biorxiv.org/content/10.1101/2024.12.19.629561v2) · Code: Alibaba DAMO PertFormer repo (mirror: [gitlink damo-pertformer](https://www.gitlink.org.cn/mirrors/damo-pertformer))
>
> PDF not committed.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context), [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed (abstract-only; bioRxiv full text rate-limited)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a 3-billion-parameter model pretrained on bulk and single-cell multi-omics (9 data types, ~1.5B paired samples derived from ~1M single cells) that predicts effects of perturbations (e.g., TF knockdown/overexpression) zero-shot. **(paper)**
- **Why it matters for the virtual cell**: it is the closest prior work to the Master's goal (multi-omic context + perturbation); any thesis claim must be positioned against it. **(my take)**
- **Main claim**: zero-shot gains of 20.9%-480.2% over existing methods; predicted tumour treatment targets validated experimentally. **(paper)**

## 2. Problem it solves

Perturbation models need perturbation training data for each context; PertFormer aims at zero-shot prediction from multi-omic pretraining. **(paper)**

## 3. Method

Multimodal transformer over paired omics (incl. chromatin), in-silico perturbation by editing inputs. Pairing construction ("1.5B paired samples from ~1M cells") suggests many derived pairs per cell. Details abstract-only. **(paper / abstract-only)**

## 4. Data

9 data types, bulk + single cell; exact datasets abstract-only.

## 5. Evaluation and results

Headline relative gains 20.9-480.2%; baselines, metrics and splits abstract-only. The wide range of relative gains suggests some baselines are weak or metrics are near zero. **(my take)**

## 6. Critical analysis

- Limitations (paper): abstract-only.
- Weaknesses I see (my take): relative-improvement headlines hide whether it beats mean / no-change / linear baselines (the known failure of scFMs in PertEval-scFM and Systema-style analyses). "Zero-shot" must be checked for leakage: perturbed genes or cell lines present in pretraining.
- What would change my mind: a Systema-corrected evaluation on unseen knockouts in an unseen cell line, beating the mean-shift baseline.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: a candidate test of the Master's H at scale. If PertFormer with chromatin input beats its RNA-only ablation on OOD knockouts, H is supported by someone else's model and my work should focus on mechanism (E4.2) and shortcut checks (E4.4). If it does not beat the mean baseline, that motivates E4.1's careful baselines. **(my take)**
- Hypotheses worth testing:
  - **H-pf1** (E4.4): PertFormer's zero-shot predictions on Norman/Replogle unseen knockouts do not beat a perturbation-mean baseline after Systema correction. *Null (for my H)*: it beats it by > CI.
  - **H-pf2** (E4.1 ablation): dropping the ATAC input from PertFormer increases OOD error. *Null*: no change.

## 8. Related work

[GEARS](gears.md), [CPA](cpa.md), [scGPT](scgpt.md), [HoloCell](holocell.md), PertEval-scFM, CellxPert.

## Resources

- Preprint: https://www.biorxiv.org/content/10.1101/2024.12.19.629561v2
