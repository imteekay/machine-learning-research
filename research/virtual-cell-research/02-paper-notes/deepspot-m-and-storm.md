# DeepSpot-M and STORM: the newest histology–transcriptomics foundation models

> (1) Nonchev K, Dawo S, Silina K, Koelzer VH, Rätsch G. *DeepSpot-M: a multimodal foundation model for transcriptome-wide virtual spatial transcriptomics from histology*. medRxiv (2026). DOI: [10.64898/2026.06.19.26356060](https://www.medrxiv.org/content/10.64898/2026.06.19.26356060v1)
> (2) Xiang J, Hou S, Li Y, …, Diehn M, Zhou X, Li R. *A multimodal foundation model of spatial transcriptomics and histology for biological discovery and clinical prediction* (STORM). arXiv (2026). arXiv: [2604.03630](https://arxiv.org/abs/2604.03630)
>
> Both read at abstract/landing-page level; no metric values were available. Code: none found for either model.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**:
  - DeepSpot-M: predicts spatial expression from H&E as a *query over genes*, with gene embeddings from DNA, RNA, protein, single-cell and text foundation models, so it can predict genes not seen in training; test-time adaptation from one slide. **(paper)**
  - STORM: hierarchical model integrating morphology, expression and spatial context, pretrained on ~1.2M spatial profiles with matched histology across 18 organs. **(paper)**
- **Why it matters for the virtual cell**: they push histology bridges from "predict 250 HVGs" to "transcriptome-wide" (DeepSpot-M) and to "clinical prediction across 7k patients" (STORM). **(my take)**
- **Main claim**: DeepSpot-M beats specialized models on held-out cancers and produced a virtual atlas of 28,664 TCGA slides (32 cancers). STORM improves spatial domain discovery, H&E→expression across 11 tumor types, and immunotherapy/prognosis over established biomarkers in 23 cohorts (7,245 patients). **(paper)**

## 2. Problem it solves

Measured ST is costly; earlier models cover a fixed gene panel and one cohort. **(paper)**

## 3. Method

- DeepSpot-M: gene-as-query decoder over image features; multi-FM gene embeddings; test-time adaptation. **(paper)**
- STORM: hierarchical spot/region/slide representation; platforms Visium, Visium HD, Xenium, CosMx. **(paper)**
- Training objectives and sizes: abstract-only.

## 4. Data

- DeepSpot-M: TCGA WSIs, HEST-1K, lung Xenium, mouse spatial total RNA-seq, WSSS4LUAD, MOSAIC Window (EGA, controlled), USZ cohorts. Released: **virtual TCGA ST atlas** and **HEST Xenium virtual ST** (Hugging Face). **(paper)**
- STORM: ~1.2M profiles, 18 organs; 23 clinical cohorts, 7,245 patients. **(paper)**

## 5. Evaluation and results

Qualitative claims only in what I read; numeric results abstract-only.

## 6. Critical analysis

- Limitations (paper): STORM marked work in progress. Others not retrieved.
- Weaknesses I see (my take):
  - "Unseen gene" prediction from gene embeddings may just predict a gene's co-expression module mean; test against a "nearest-seen-gene" baseline.
  - Virtual TCGA atlases risk being used as ground truth downstream; errors become silent confounders.
  - Neither reports, in what I read, a fixed-encoder comparison (STP-BENCH shows encoder choice reorders rankings).
- What would change my mind: per-gene gains on STP-BENCH with a shared encoder.

## 7. Relevance to my research

- **Research proposal**: a virtual TCGA atlas is a cheap source of spatial context for PhD pretraining, if accuracy holds near tumor borders. **(my take)**
- **Roadmap**: E6.2 (DeepSpot-M is a named candidate), E6.1 (STORM-style clinical prediction), E5.3.
- Hypotheses worth testing:
  - **H-DS1 (E6.2)**: for held-out genes, DeepSpot-M beats a "predict from the most similar trained gene (by embedding)" baseline by > 0.05 PCC. *Null*: no gain.
  - **H-DS2**: pretraining a niche model on virtual TCGA ST then fine-tuning on real Xenium improves E5.1 niche-prediction over real-data-only training. *Null*: no gain or harm.

## 8. Related work

DeepSpot (earlier version, same lab), [HEX](hex-virtual-spatial-proteomics.md) (same group as STORM), [STPath](stpath.md), [OmiCLIP/Loki](omiclip-loki.md), [STP-BENCH](stp-bench-and-st-prediction-review.md).

## Resources

- [TCGA virtual ST atlas](https://huggingface.co/datasets/ratschlab/TCGA_virtual_spatial_transcriptomics_atlas) · [HEST Xenium virtual ST](https://huggingface.co/datasets/ratschlab/HEST_Xenium_virtual_spatial_transcriptomics)
