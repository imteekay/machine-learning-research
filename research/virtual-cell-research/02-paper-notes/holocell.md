# HoloCell: one generative model over chromatin, RNA and protein

> Jiang Q, Li Z, Hu B, ..., Yin Q. *HoloCell: A Generative Foundation Model for Holistic Cellular Modeling*. bioRxiv (2026). DOI: [10.64898/2026.06.07.730684](https://www.biorxiv.org/content/10.64898/2026.06.07.730684v1) · Code: not checked
>
> PDF not committed.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: skimmed (abstract-only; bioRxiv full text rate-limited)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: an 860M+ parameter generative model pretrained on the "Human-Multi-Omics-Corpus" (~468M single-cell profiles, >425B tokens) that tokenises cis-regulatory elements, genes and proteins in one hierarchical vocabulary and generates with iterative diffusion-style masking/remasking. **(paper)**
- **Why it matters for the virtual cell**: it is the most direct large-scale instance of the Master's architecture (shared latent over ATAC+RNA, generative). It will be the "scale" baseline my context-focused hypothesis must beat or explain. **(my take)**
- **Main claim**: superior performance on single-omics representation, paired integration, unpaired alignment and cross-modal generation; generation order is not fixed, so it can "simulate information flow across omics layers". **(paper)**

## 2. Problem it solves

Existing scFMs are single-modality or pairwise; no model jointly represents and generates epigenome, transcriptome and proteome. **(paper)**

## 3. Method

- Hierarchical tokens: CREs (peaks) → genes → proteins in one framework. **(paper)**
- Masked discrete diffusion with remasking for any-to-any generation. **(paper)**
- Details of how peaks are linked to genes, and context length: abstract-only.

## 4. Data

Human-Multi-Omics-Corpus: ~468M single-cell profiles across three omics; composition (how much is paired vs single-modality) abstract-only. Potentially a new dataset/corpus. **(paper)**

## 5. Evaluation and results

No numbers in the abstract. Perturbation prediction is not listed among evaluated tasks. **(paper / abstract-only)**

## 6. Critical analysis

- Limitations (paper): abstract-only.
- Weaknesses I see (my take): "468M profiles" are probably mostly RNA-only, so the multi-omic part may be small; the claim of simulated information flow is not causal unless tested with perturbations. Cross-modal generation metrics often reward cell-type recovery.
- What would change my mind: ATAC→RNA generation that beats a cell-type-mean baseline within types, and an in-silico TF knockout (mask CREs with a motif) that matches a Perturb-multiome result.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: it tests the "scale" side of "context not scale". If HoloCell's ATAC conditioning does not reduce OOD perturbation error, that weakens the Master's H at scale; if it does, my contribution must be a cleaner causal evaluation (E4.1, E4.4). Its unpaired alignment is a candidate module for E3.2. **(my take)**
- Hypotheses worth testing:
  - **H-holo1**: HoloCell RNA-conditioned-on-ATAC predictions improve more for TF-driven, cell-type-specific genes than housekeeping genes (E3.1 design applied to a pretrained model). *Null*: uniform gain.
  - **H-holo2**: zeroing accessibility at a TF's motif sites in HoloCell input shifts predicted RNA of that TF's known targets (E0.2-style target sets) more than random gene sets. *Null*: no target enrichment.

## 8. Related work

[scGPT](scgpt.md), [CAPTAIN](captain.md), [PertFormer](pertformer.md), Lingshu-Cell, scCLIP ([note](scclip.md)).

## Resources

- Preprint: https://www.biorxiv.org/content/10.64898/2026.06.07.730684v1
