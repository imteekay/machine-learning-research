# DeepSpot-M and STORM: the newest histology–transcriptomics foundation models

> (1) Nonchev K, Dawo S, Silina K, Koelzer VH, Rätsch G. *DeepSpot-M: a multimodal foundation model for transcriptome-wide virtual spatial transcriptomics from histology*. medRxiv (2026). DOI: [10.64898/2026.06.19.26356060](https://www.medrxiv.org/content/10.64898/2026.06.19.26356060v1)
> (2) Xiang J, Hou S, Li Y, …, Diehn M, Zhou X, Li R. *A multimodal foundation model of spatial transcriptomics and histology for biological discovery and clinical prediction* (STORM). arXiv (2026). arXiv: [2604.03630](https://arxiv.org/abs/2604.03630)
>
> DeepSpot-M read at abstract/landing-page level (no metric values). STORM full text read via arXiv HTML on 2026-11-25. Code: none found for either model.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed (STORM: full text read; DeepSpot-M: abstract-only)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**:
  - DeepSpot-M: predicts spatial expression from H&E as a *query over genes*, with gene embeddings from DNA, RNA, protein, single-cell and text foundation models, so it can predict genes not seen in training; test-time adaptation from one slide. **(paper)**
  - STORM: hierarchical model integrating morphology, expression and spatial context, pretrained on ~1.2M spatial profiles with matched histology across 18 organs. **(paper)**
- **Why it matters for the virtual cell**: they push histology bridges from "predict 250 HVGs" to "transcriptome-wide" (DeepSpot-M) and to "clinical prediction across 7k patients" (STORM). **(my take)**
- **Main claim**: DeepSpot-M beats specialized models on held-out cancers and produced a virtual atlas of 28,664 TCGA slides (32 cancers). STORM improves spatial domain discovery, H&E→expression across 11 tumor types, and immunotherapy/prognosis over established biomarkers in 23 cohorts (7,245 patients). **(paper)**

## Research summary

- **Context (already known)**: H&E histology can predict spatial gene expression for fixed gene panels in single cohorts, while measured spatial transcriptomics is costly and low-throughput. **(paper)**
- **Gap (not known)**: Whether histology models can reach transcriptome-wide or unseen-gene prediction (DeepSpot-M), and whether joint morphology-expression pretraining improves spatial and clinical prediction (STORM). **(paper)**
- **Hypothesis / what they try to determine**: DeepSpot-M: a gene-as-query decoder with gene embeddings from several foundation models predicts unseen genes. STORM: a hierarchical model that integrates morphology, expression and spatial context outperforms histology-only models. **(paper)**
- **Approach (how they investigated)**: DeepSpot-M: abstract only (TCGA, HEST-1K, Xenium and other sets). STORM (full text read): spot-level encoders plus a 5x5 neighbourhood spatial encoder, self-supervised masked pretraining on ~1.2M spots from 632 paired sections, benchmarked on domain discovery, expression prediction and prognosis. **(paper)**
- **Results (what they found)**: STORM: kidney-cancer domain NMI/ARI 0.53/0.40 vs OmiCLIP 0.41/0.32; median PCC 0.633 in colorectal vs DeepPT 0.304 and Virchow 0.266 (500 HVGs, 8 datasets); prognosis C-index 0.701-0.741 on two test cohorts. DeepSpot-M: claims to beat specialized models on held-out cancers; no numbers seen. **(paper)**
- **Implications (what it means)**: Both move histology-to-expression toward clinical use, but DeepSpot-M's unseen-gene claim and any virtual atlas need baselines before use as ground truth. STORM's own text gives inconsistent cohort counts (11 cohorts, >2,500 patients in the Discussion vs 23 and 7,245 in the abstract), so check before citing. **(my take)**

## 2. Problem it solves

Measured ST is costly; earlier models cover a fixed gene panel and one cohort. **(paper)**

## 3. Method

- DeepSpot-M: gene-as-query decoder over image features; multi-FM gene embeddings; test-time adaptation. **(paper)**
- STORM: two-level design. Spot level: pretrained encoders for H&E patches and for ST (NOVAE embeddings); spatial level: features from a 5x5 neighbourhood of spots processed jointly. Self-supervised multimodal masked pretraining reconstructs masked spots. Platforms Visium, Visium HD, Xenium, CosMx. **(paper)**
- DeepSpot-M training objective and sizes: abstract-only.

## 4. Data

- DeepSpot-M: TCGA WSIs, HEST-1K, lung Xenium, mouse spatial total RNA-seq, WSSS4LUAD, MOSAIC Window (EGA, controlled), USZ cohorts. Released: **virtual TCGA ST atlas** and **HEST Xenium virtual ST** (Hugging Face). **(paper)**
- STORM: ~1.2M spots from 632 paired H&E-ST sections, 18 organs; expression-prediction benchmark on 8 datasets (142 sections, 85,236 spots, 8 tumor types); clinical prediction cohorts (abstract: 23 cohorts, 7,245 patients; Discussion says 11 cohorts, >2,500 patients: unresolved). **(paper)**

## 5. Evaluation and results

STORM **(paper)**: domain discovery on kidney and lung Visium: KC NMI/ARI 0.53/0.40 vs OmiCLIP 0.41/0.32, LC 0.62/0.60 vs 0.58/0.49. Expression prediction (PCC on 500 HVGs; baselines DeepPT, TCGN, GigaPath, UNI, Virchow, OmiCLIP): highest median PCC in all 8 tumor types, e.g. colorectal 0.633 vs DeepPT 0.304 and Virchow 0.266. Prognosis (DSS C-index, SURGEN colorectal and NLST lung test cohorts): 0.701-0.741, best among compared models. Immunotherapy numbers not extracted. DeepSpot-M: numeric results abstract-only.

## 6. Critical analysis

- Limitations (paper): STORM is marked a work-in-progress preprint and has no limitations section. DeepSpot-M: not retrieved.
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
