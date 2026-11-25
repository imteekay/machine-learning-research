# STP-BENCH + "Predicting spatial transcriptomics from histology" review: how well virtual ST works today

> (1) Chung Y, Ha JH, Song AH, Almagro-Pérez C, Seo C, Suh WJ, Beom JW, Oh KB, Ruppin E, Mahmood F, Lee JS. *STP-BENCH: Benchmark for Virtual Spatial Transcriptomics from Histopathology Images*. arXiv (2026). arXiv: [2609.05956](https://arxiv.org/abs/2609.05956) · Code: [NEXGEM/STP-Bench](https://github.com/NEXGEM/STP-Bench)
> (2) Xu K, Koohy H, Antanaviciute A. *Predicting spatial transcriptomics from histology images: progress, challenges, and directions*. EXO (2026). [EXO.2026.0020](https://www.sciexplor.com/exo/articles/EXO.2026.0020)
>
> (1) read at abstract level (no numbers on the page). (2) review read in full text (first ~100k characters).

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: (1) a standardized benchmark of 21 H&E→ST methods over 6 cancer types and Visium + Xenium, with a shared pathology-FM encoder; (2) a review of > 40 H&E→ST models. **(paper)**
- **Why it matters for the virtual cell**: they set the bar E6.2 must use and list the leakage traps. **(my take)**
- **Main claim**: (1) a shared image encoder substantially reorders model rankings: earlier papers conflated architecture with encoder. (2) Typical Pearson is 0.0–0.6; models are limited by data quantity/diversity; context models (GNN/ViT) have not clearly beaten CNNs. **(paper)**

## 2. Problem it solves

Inconsistent datasets, gene sets, normalization and splits make H&E→ST results incomparable. **(paper)**

## 3. Method

- (1) Evaluates per-gene and gene-set recoverability, downstream deconvolution and spatial domains, robustness to domain shift and data scaling; models re-implemented with one pathology FM where possible. Each training set > 30k spots and ≥ 15 slides. **(paper)**
- (2) Comparison of data, modeling, evaluation and translation; proposes a minimal reporting standard. **(paper)**

## 4. Data

- (1) 6 cancer types, Visium + Xenium. **(paper)**
- (2) Corpora: HEST-1k, STimage-1K4M, ST-bank, ViSTomics-4M. Total training across the field ≈ 750 sections, most models < 100, skewed to breast cancer and Visium. **(paper)**

## 5. Evaluation and results

- (1) Ranking reshuffle under shared encoding; numbers: abstract-only.
- (2) PCC ~0.0–0.6; Xenium predicted better than Visium; leakage via shared donors, serial sections, pre-split batch correction. Open: are models capturing cell states or only composition? **(paper)**

## 6. Critical analysis

- Limitations (paper): (2) notes non-cancer diseases unexplored and little orthogonal validation.
- Weaknesses I see (my take): neither (as far as I read) evaluates *patient-level biomarker ranking*, which is the clinically relevant quantity in E6.2.
- What would change my mind: a benchmark showing patient-level biomarker Spearman is as poor as spot-level PCC.

## 7. Relevance to my research

- **Research proposal**: (2)'s "state vs composition" question is the core of the PhD's histology use: if H&E only gives composition, it can't replace measured spatial data for perturbation response. **(my take)**
- **Roadmap**: E6.2 should adopt STP-BENCH splits and the shared-encoder rule; E4.4 (shortcut) logic applies: composition is the shortcut.
- Hypotheses worth testing:
  - **H-SB1 (E6.2 + E4.4)**: after regressing out predicted cell-type composition per spot, residual predicted expression has PCC < 0.1 with residual measured expression. If so, models capture composition, not state.
  - **H-SB2 (E6.2)**: patient-level biomarker Spearman ranking of methods differs from the STP-BENCH spot-level ranking (Kendall τ < 0.5).
  - **H-SB3**: a donor-level split lowers reported PCC by > 0.05 vs a slide-level split on HEST (quantify leakage).

## 8. Related work

HEST-1K, [STPath](stpath.md), [DeepSpot-M / STORM](deepspot-m-and-storm.md), [OmiCLIP/Loki](omiclip-loki.md), [HEX](hex-virtual-spatial-proteomics.md).

## Resources

- [STP-Bench code](https://github.com/NEXGEM/STP-Bench)
