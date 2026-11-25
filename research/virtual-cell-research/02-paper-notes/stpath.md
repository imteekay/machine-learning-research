# STPath: a generative foundation model for spatial transcriptomics from whole-slide images

> Huang T, Liu T, Babadi M, Ying R, Jin W. *STPath: a generative foundation model for integrating spatial transcriptomics and whole-slide images*. npj Digital Medicine (2025). DOI: [10.1038/s41746-025-02020-3](https://www.nature.com/articles/s41746-025-02020-3) · Preprint: [bioRxiv 2025.04.19.649665](https://www.biorxiv.org/content/10.1101/2025.04.19.649665) · Weights: [tlhuang/STPath](https://huggingface.co/tlhuang/STPath)
>
> Journal page behind a login redirect; numbers are from the preprint abstract via search results.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a slide-level generative model trained on paired H&E + spatial transcriptomics that predicts expression for any spot of a whole-slide image, across organs and platforms, without per-dataset fine-tuning. **(paper)**
- **Why it matters for the virtual cell**: it is one of the candidate models for E6.2 (histology → spatial expression → patient biomarkers). **(my take)**
- **Main claim**: average Pearson 0.266 vs 0.198 for the slide-based TRIPLEX baseline (+34%), and gains across 5 tasks. **(paper)**

## 2. Problem it solves

Earlier H&E→ST models are trained per dataset and per gene panel. STPath is pretrained once across organs, genes and technologies. **(paper)**

## 3. Method

Pathology-FM patch features over the whole slide + a generative (masked) objective over spot expression with spatial context; supports gene expression prediction and spot imputation from partially measured slides. Architecture details: abstract-only. **(paper)**

## 4. Data

HEST-1K + STimage-1K4M: 983 slides, 38,984 genes, 17 organs, 4 sequencing technologies. **(paper)**

## 5. Evaluation and results

- 5 tasks across 23 datasets and 9 biomarkers: expression prediction, spot imputation, spatial clustering, mutation prediction, survival prediction. **(paper)**
- Mean PCC 0.266 vs 0.198 (TRIPLEX). **(paper)** Other numbers: abstract-only.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take):
  - Mean PCC of 0.27 is mostly driven by highly variable, morphology-linked genes (immune, stromal). Per-gene breakdown matters more than the mean.
  - Serial sections / same donor across HEST splits can inflate results (EXO review warns about this).
  - TRIPLEX is a reasonable but not the strongest baseline; STP-BENCH shows the image encoder choice reorders rankings, so the gain may partly be the encoder.
- What would change my mind: per-gene gains that hold on a donor-held-out, platform-shifted cohort with the same encoder for all methods.

## 7. Relevance to my research

- **Research proposal**: the PhD's tissue model needs spatial data; if H&E can stand in, the model can run on archived clinical slides. **(my take)**
- **Roadmap**: E6.2 (biomarker rankings from predicted maps), E5.3 (generalization to unseen tissues).
- Hypotheses worth testing:
  - **H-STP1 (E6.2)**: patient-level immune-infiltration scores from STPath maps correlate with measured ones (Spearman > 0.6) even though gene-level PCC is ~0.27. *Null*: Spearman CI includes 0.3.
  - **H-STP2**: STPath with the same pathology FM as a linear-probe baseline gives < 0.02 PCC gain. If so, the gain is the encoder, not the generative model.

## 8. Related work

TRIPLEX, BLEEP, [OmiCLIP/Loki](omiclip-loki.md), [DeepSpot-M / STORM](deepspot-m-and-storm.md), [STP-BENCH](stp-bench-and-st-prediction-review.md), HEST-1K.

## Resources

- [Hugging Face README](https://huggingface.co/tlhuang/STPath/blob/main/README.md)
