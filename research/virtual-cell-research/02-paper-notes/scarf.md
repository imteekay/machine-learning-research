# SCARF: joint scRNA + scATAC foundation model and the X-Omics 2.7M-cell corpus

> Liu G. et al. *SCARF: Single Cell ATAC-seq and RNA-seq Foundation model*. bioRxiv (2025). DOI: [10.1101/2025.04.07.647689](https://doi.org/10.1101/2025.04.07.647689) · Code: not retrieved
>
> bioRxiv returned 429. These notes are abstract-level only.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: skimmed (abstract-only)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a foundation model pretrained jointly on scRNA and scATAC from **X-Omics**, a corpus of more than 2.7M cells across tissues and species. **(paper)**
- **Why it matters for the virtual cell**: joint pretraining is the obvious alternative to "separate encoders + alignment" for the Master's model. X-Omics may be the largest paired or unpaired RNA+ATAC pretraining set available. **(my take)**
- **Main claim**: state-of-the-art results on cell representation, cross-modality cell matching, and cross-omics translation. **(paper)**

## Research summary

- **Context (already known)**: scRNA and scATAC foundation models mostly exist as unimodal models, and paired multiome data carries complementary information. **(paper)**
- **Gap (not known)**: Whether joint pretraining on RNA and ATAC at multi-million-cell scale improves representation, cross-modal matching and translation; details are not available in the abstract. **(my take)**
- **Hypothesis / what they try to determine**: Joint pretraining on the X-Omics corpus (>2.7M cells, multiple tissues and species) gives better cell representation, cross-modality matching and cross-omics translation. **(paper)**
- **Approach (how they investigated)**: Pretrain a joint RNA+ATAC model on X-Omics and evaluate on representation, cell matching and translation tasks. Specifics not available in abstract (bioRxiv returned HTTP 429). **(paper, abstract-level)**
- **Results (what they found)**: Claims state-of-the-art on the three task types; no numbers available in abstract. **(paper)**
- **Implications (what it means)**: If matching holds within cell type it is a strong baseline and corpus for E3.2; unverified until full text and baselines are read. **(my take)**

## 2. Problem it solves

Unimodal FMs cannot use the complementary information in paired multiome data. **(paper)**

## 3. Method

Abstract-only. Details were not retrieved.

## 4. Data

**X-Omics**: more than 2.7M cells, multiple tissues and species, RNA and ATAC. **(paper)** This is a candidate entry for datasets.md. Availability was not verified.

## 5. Evaluation and results

Abstract-only. Numbers were not retrieved.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): "cell matching" scores depend heavily on how many cells share a type, so FOSCTTM within cell type is the honest metric. It is unclear how much of X-Omics is truly paired.
- What would change my mind: within-cell-type FOSCTTM that clearly beats a simple contrastive baseline (scCLIP) trained on the same paired data.

## 7. Relevance to my research

- **Hypotheses**: a direct competitor or baseline for E3.2 (cross-modal matching), and a possible pretraining corpus for the Master's model.
- **Hypotheses worth testing**:
  - H-SC1 (E3.2): joint pretraining beats separately pretrained encoders plus a CLIP head (Atacformer-CRAFT) on held-out pairing FOSCTTM. *Null*: no difference.
  - H-SC2: SCARF-imputed ATAC for RNA-only Perturb-seq cells gives the same E4.1 gain as measured ATAC, within 50%. *Null*: imputed ATAC gives no gain.

## 8. Related work

[EpiFoundation](epifoundation.md), [CLM-X](clm-x.md), [scLinguist](sclinguist.md), scCLIP, GLUE, MultiVI.

## Resources

- bioRxiv: https://www.biorxiv.org/content/10.1101/2025.04.07.647689v1
