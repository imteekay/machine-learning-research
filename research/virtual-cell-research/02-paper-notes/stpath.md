# STPath: a generative foundation model for spatial transcriptomics from whole-slide images

> Huang T, Liu T, Babadi M, Ying R, Jin W. *STPath: a generative foundation model for integrating spatial transcriptomics and whole-slide images*. npj Digital Medicine (2025). DOI: [10.1038/s41746-025-02020-3](https://www.nature.com/articles/s41746-025-02020-3) · Preprint: [bioRxiv 2025.04.19.649665](https://www.biorxiv.org/content/10.1101/2025.04.19.649665) · Weights: [tlhuang/STPath](https://huggingface.co/tlhuang/STPath)
>
> Full text read via Europe PMC (PMC12618518). Supplement not read.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: read (full text via Europe PMC)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a slide-level generative model trained on paired H&E + spatial transcriptomics that predicts expression for any spot of a whole-slide image, across organs and platforms, without per-dataset fine-tuning. **(paper)**
- **Why it matters for the virtual cell**: it is one of the candidate models for E6.2 (histology → spatial expression → patient biomarkers). **(my take)**
- **Main claim**: average Pearson 0.266 vs 0.198 for the slide-based TRIPLEX baseline (+34%), and gains across 5 tasks. **(paper)**

## Research summary

- **Context (already known)**: H&E-to-ST models are trained per dataset and gene panel. **(paper)**
- **Gap (not known)**: Whether one pretrained model can predict expression across organs, genes and technologies without fine-tuning. **(paper)**
- **Hypothesis / what they try to determine**: That a generative slide-level foundation model integrating WSIs and ST generalizes broadly. **(paper)**
- **Approach (how they investigated)**: Pathology-FM patch features with a masked generative objective, trained on HEST-1K + STimage-1K4M (983 slides) and tested on 5 tasks over 23 datasets. Spots are tokens; image (Gigapath patch encoder), gene, organ and sequencing-technology tracks are summed, and an E(2)-invariant geometry-aware Transformer with spatially biased attention predicts masked expression with a regression loss. **(paper)**
- **Results (what they found)**: Pretraining-validation PCC (top 200 HVGs) 0.266 vs 0.198 for TRIPLEX (+34.4%). On HEST-Bench (zero-shot, top 50 HVGs) it ranks first in PCC on 8 of 10 datasets and in AMI on 9 of 10, beating the next best by 6.9% PCC (UNI) and 14.4% AMI (TRIPLEX); PCC 0.75 on IDC, 0.60 on SKCM, 0.559 on LUNG. **(paper)**
- **Implications (what it means)**: Promising, but the gain may be encoder-driven and PCC of 0.27 is low; donor-held-out per-gene tests are needed. **(my take)**

## 2. Problem it solves

Earlier H&E→ST models are trained per dataset and per gene panel. STPath is pretrained once across organs, genes and technologies. **(paper)**

## 3. Method

Each spot is a token: Gigapath patch-encoder features (linear-projected), log1p gene expression over a 38,984-symbol vocabulary aggregated by a linear layer (a weighted bag of gene embeddings), plus organ and technology embeddings, all summed. A spatial Transformer biases attention with coordinates through invariant frame averaging (E(2)-invariant). Masked genes are predicted with a regression loss (not classical masked-LM) under tailored noise/masking schedules. Prompting a few measured spots (0, 5, 10, 20%) gives in-context imputation. **(paper)**

## 4. Data

HEST-1K + STimage-1K4M, de-duplicated and with HEST-Bench and clustering samples removed: 983 WSIs, 38,984 genes, 17 organs; 928 slides for pretraining and 5% held out for validation. **(paper)**

## 5. Evaluation and results

- Six tasks across 23 datasets and 14 biomarkers: expression prediction, spot imputation, spatial clustering, biomarker prediction, mutation prediction, survival prediction. **(paper)**
- Baselines (UNI, Gigapath, BLEEP, TRIPLEX) trained on the same cohort; evaluated zero-shot per dataset rather than leave-one-out. Significance by two-sided Wilcoxon, p<0.001. **(paper)**
- Predicted expression correlates best with expert annotations (brain layers, tumor regions) across five datasets; TP53 mutation AUC 0.73 in breast cancer. **(paper)**

## 6. Critical analysis

- Limitations (paper): no explicit limitations section found in the text I read; the Discussion calls the direction promising but underexplored. **(paper)**
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
