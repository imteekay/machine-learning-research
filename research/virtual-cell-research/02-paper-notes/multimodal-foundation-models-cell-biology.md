# Multimodal foundation models in molecular cell biology: a perspective on combining omics modalities and scales

> Cui H, Tejada-Lapuerta A, Brbić M, Saez-Rodriguez J, Cristea S, Goodarzi H, Lotfollahi M, Theis FJ, Wang B. *Towards multimodal foundation models in molecular cell biology*. Nature (2025). DOI: [10.1038/s41586-025-08710-y](https://doi.org/10.1038/s41586-025-08710-y) · Code: none (perspective) · [Video](https://www.youtube.com/watch?v=7LW9Pn7-YtE)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 0](../00-plan/roadmap.md#phase-0-foundations), [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a perspective proposing multimodal foundation models pretrained jointly on genomics, transcriptomics, epigenomics, proteomics, metabolomics and spatial data. **(paper)**
- **Why it matters for the virtual cell**: it is the map of which modalities exist and how they could share one model. The Master's project (RNA + ATAC) is the smallest real case of this vision. **(my take)**
- **Main claim**: like LLMs in NLP, joint pretraining on diverse omics would give holistic maps of cells, genes and tissues and support transfer to cell-type discovery, biomarkers, gene regulation inference and in silico perturbation. **(paper)**

## Research summary

- **Context (already known)**: omics data are growing quickly and foundation models trained on single modalities already exist. **(paper)**
- **Gap (not known)**: how to build models that combine genomics, transcriptomics, epigenomics, proteomics, metabolomics and spatial data. **(paper)**
- **Hypothesis / what they try to determine**: joint multimodal pretraining, like LLMs in NLP, would give holistic maps of cells and support transfer to downstream tasks. **(paper)**
- **Approach (how they investigated)**: a perspective: it reviews modalities and scales and sets out a vision; no experiments. **(paper)**
- **Results (what they found)**: abstract-only: a case for multimodal pretraining and context-specific transfer; details not read (Nature paywall). **(paper)**
- **Implications (what it means)**: it frames the Master's RNA+ATAC project as the smallest real case, though paired data scarcity and weak perturbation benchmarks temper the scale argument. **(my take)**

## 2. Problem it solves

Omics data grows faster than our ability to analyze it, and most models handle one modality. The authors argue a joint model is needed to describe cell state across a continuum. **(paper)**

## 3. Method

Abstract-only: I could not access the full text (Nature paywall). From the abstract: pretrain on many omics, then do context-specific transfer learning for downstream tasks. **(paper)** Specific architecture recommendations (tokenization across modalities, alignment objectives) not checked.

## 4. Data

No new dataset. Abstract-only for the list of resources discussed.

## 5. Evaluation and results

Perspective, no experiments. Abstract-only for any evaluation recommendations.

## 6. Critical analysis

- Limitations (paper): abstract-only; not checked.
- Weaknesses I see (my take):
  - The LLM analogy assumes scale and diversity are the bottleneck. Benchmarks on perturbation prediction suggest that current scFMs do not beat simple baselines, so scale alone may not be enough.
  - Paired multimodal data (same cell, several modalities) is rare. Most "multimodal" pretraining will be on unpaired data, where alignment can hallucinate correspondences.
- What would change my mind: evidence that adding a second modality during pretraining improves a *single-modality* downstream task (e.g. RNA-only perturbation prediction) would show real cross-modal transfer.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: the Master's hypothesis is a narrow, testable version of this vision: does adding ATAC to an RNA model reduce OOD perturbation error (E4.1)? Phase 3 (E3.1, E3.2) checks the preconditions: that ATAC carries regulatory information not in RNA, and that cells can be matched across modalities.
- Hypotheses worth testing:
  - **H-MM-1 (paired vs unpaired)**: an RNA+ATAC model pretrained on paired Multiome cells beats one aligned on unpaired RNA and ATAC data in E4.1. *Null*: no difference, which would mean unpaired atlases are enough.
  - **H-MM-2 (cross-modal transfer)**: pretraining with ATAC, then fine-tuning and testing on RNA only, still lowers unseen-knockout error compared with RNA-only pretraining. If yes, ATAC acts as a training signal even where it is not measured.

## 8. Related work

- [AIVC manifesto](ai-virtual-cell-priorities.md), [Transformers in single-cell omics](transformers-single-cell-review.md) (same group), [scGPT](scgpt.md), [Nicheformer](nicheformer.md).

## Resources

- DOI: https://doi.org/10.1038/s41586-025-08710-y
- Video: https://www.youtube.com/watch?v=7LW9Pn7-YtE
