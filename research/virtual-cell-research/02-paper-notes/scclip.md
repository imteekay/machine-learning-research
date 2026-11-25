# scCLIP: contrastive RNA↔ATAC pretraining for paired single cells

> Xiong L, Chen T, Kellis M. *scCLIP: Multi-modal Single-cell Contrastive Learning Integration Pre-training*. NeurIPS 2023 Workshop. [OpenReview](https://openreview.net/pdf?id=KMtM5ZHxct) · Code: anonymised 4open.science link in paper (not checked)
>
> PDF not committed.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: skimmed (abstract-only; OpenReview blocked automated fetch)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: CLIP-style contrastive learning between a transformer over scATAC peaks and a transformer over scRNA genes, trained on paired multiome cells, giving one shared embedding. **(paper)**
- **Why it matters for the virtual cell**: it is the simplest strong baseline for E3.2 (matching cells across modalities) and for imputing ATAC context onto RNA-only Perturb-seq cells. **(my take)**
- **Main claim**: outperforms competing integration methods and transfers zero-shot to unseen datasets across modalities. **(paper)**

## Research summary

- **Context (already known)**: Paired multiome data exists, and CLIP-style contrastive learning aligns two modalities in one embedding space. **(paper)**
- **Gap (not known)**: Whether contrastive pretraining of RNA and ATAC transformers gives embeddings that integrate and transfer zero-shot across modalities and datasets. **(paper)**
- **Hypothesis / what they try to determine**: Contrastive alignment of paired RNA and ATAC cells yields a shared space usable on new unpaired data. **(paper)**
- **Approach (how they investigated)**: Two modality-specific transformers trained with InfoNCE on paired multiome cells; compared with integration methods. Details abstract-only. **(paper)**
- **Results (what they found)**: Outperforms competing integration methods and transfers zero-shot; numbers not available in abstract. **(paper)**
- **Implications (what it means)**: A simple baseline for E3.2 and ATAC imputation, but it may align only shared (cell-identity) signal. **(my take)**

## 2. Problem it solves

Integrating paired RNA+ATAC so that new unpaired data from either modality can be embedded in the same space. **(paper)**

## 3. Method

Two modality-specific transformer encoders (ATAC peaks are tokenised, likely in patches given ~100k+ peaks), InfoNCE contrastive loss over matched pairs in a batch. **(paper; tokenisation details abstract-only)**

## 4. Data

Paired 10x Multiome datasets; specifics abstract-only.

## 5. Evaluation and results

Integration and cross-modal transfer metrics vs. other methods; numbers abstract-only.

## 6. Critical analysis

- Limitations (paper): abstract-only.
- Weaknesses I see (my take): contrastive objectives align what is *shared* between modalities (mostly cell identity) and can discard ATAC-specific regulatory information, which is the information the Master's H needs. Batch-level negatives from the same cell type are near-duplicates, so FOSCTTM may plateau.
- What would change my mind: scCLIP ATAC embeddings that carry information about RNA *beyond* cell type (E3.1 style residual test).

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: E3.2 explicitly lists "scCLIP-style contrastive" as one arm. It is also the likely ATAC-imputation module feeding E4.1. **(my take)**
- Hypotheses worth testing:
  - **H-clip1** (E3.2): scCLIP-style matching beats gene-activity kNN on FOSCTTM on held-out multiome cells. *Null*: no better.
  - **H-clip2**: imputed ATAC from a contrastive model carries less regulatory information than measured ATAC: in E3.1, the TF-specific gain shrinks when measured ATAC is replaced with scCLIP-retrieved ATAC. *Null*: same gain, meaning imputation is good enough for E4.1.

## 8. Related work

CLIP, GLUE, MultiVI, [Labeled GW-OT](labeled-gw-ot-cross-modality-perturbation.md), [HoloCell](holocell.md), scPairing.

## Resources

- Paper: https://openreview.net/pdf?id=KMtM5ZHxct
- NeurIPS listing: https://neurips.cc/virtual/2023/75765
