# scLinguist: Hyena-based pretrained cross-modality translation (mainly RNA to protein)

> *scLinguist: A pre-trained hyena-based foundation model for cross-modality translation in single-cell multi-omics*. bioRxiv (2025). DOI: [10.1101/2025.09.30.679123](https://doi.org/10.1101/2025.09.30.679123) · Docs: [sclinguist.readthedocs.io](https://sclinguist.readthedocs.io)
>
> bioRxiv full text still returned 429 on retry. These notes come from the abstract, the docs overview and the bioRxiv API metadata. Authors (API metadata): Fang Z, Miao Z, Lin J, Xie Y, Tang J, Ding J, Li M (corresponding: Min Li, Central South University).

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: skimmed (abstract-only; full text blocked on retry)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: an encoder-decoder with stacked Hyena (long-convolution) blocks, trained in 3 stages: self-supervised pretraining on large unimodal single-cell data, then paired fine-tuning for translation. The main use is RNA to protein (CITE-seq). **(paper)**
- **Why it matters for the virtual cell**: Hyena scales subquadratically to all genes or peaks, which is relevant if the Master's model must take whole-genome ATAC. Pretraining on unpaired data before translating helps when paired data is scarce. **(my take)**
- **Main claim**: better cross-modality translation, especially RNA to protein, than existing methods. **(paper)**

## Research summary

- **Context (already known)**: Paired multi-modal data (CITE-seq) is scarce while unimodal single-cell data is abundant, and RNA-to-protein translation tools exist (BABEL, totalVI). **(paper)**
- **Gap (not known)**: Whether pretraining on large unimodal data, then fine-tuning on paired data, improves cross-modality translation. **(paper)**
- **Hypothesis / what they try to determine**: A Hyena-based encoder-decoder with staged pretraining translates modalities better than current methods, especially RNA to protein. **(paper)**
- **Approach (how they investigated)**: Three-stage training: unimodal self-supervised pretraining, then paired fine-tuning (abstract/snippet-level). **(paper)**
- **Results (what they found)**: The abstract claims better cross-modality translation, mainly RNA to protein; no numbers were retrieved. **(paper)**
- **Implications (what it means)**: If real, unimodal data can substitute for scarce paired data; this remains unverified without full text and baseline details. **(my take)**

## 2. Problem it solves

Paired multi-omics data is scarce, and full-length feature sequences are too long for attention. **(paper)**

## 3. Method

Abstract-only. The docs overview describes a modality-specific pretraining then fine-tuning on paired RNA-protein data, and a "transformer-based, two-step" framework, which disagrees with the preprint's Hyena / 3-stage description. **(code vs paper)**

## 4. Data

Large unimodal single-cell datasets plus paired CITE-seq. Details not retrieved.

## 5. Evaluation and results

Abstract-only. No numbers retrieved. Expected baselines are BABEL, scButterfly, and totalVI.

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): RNA to protein is easier than RNA to ATAC, because protein mostly tracks its own mRNA. That makes it a weak test of regulatory translation.
- What would change my mind: strong RNA to ATAC results evaluated per peak on distal peaks, not only promoter peaks.

## 7. Relevance to my research

- **Hypotheses**: candidate translator for E3.2, used to impute ATAC for RNA-only screens. It is also an architecture option (Hyena vs attention) for long peak sequences.
- **Hypotheses worth testing**:
  - H-SL1: on RNA to ATAC, a Hyena decoder matches attention on promoter peaks but is better on distal peaks, because it has the context to reach them. *Null*: no difference.
  - H-SL2: unimodal pretraining reduces the paired data needed to reach a fixed translation PCC by at least 2x. *Null*: the learning curves overlap.

## 8. Related work

BABEL, scButterfly, totalVI, HyenaDNA, [SCARF](scarf.md), [CLM-X](clm-x.md).

## Resources

- bioRxiv: https://www.biorxiv.org/content/10.1101/2025.09.30.679123v1
- Docs: https://sclinguist.readthedocs.io
