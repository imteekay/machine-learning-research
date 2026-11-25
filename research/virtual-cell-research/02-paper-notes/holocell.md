# HoloCell: one generative model over chromatin, RNA and protein

> Jiang Q, Li Z, Hu B, ..., Yin Q. *HoloCell: A Generative Foundation Model for Holistic Cellular Modeling*. bioRxiv (2026). DOI: [10.64898/2026.06.07.730684](https://www.biorxiv.org/content/10.64898/2026.06.07.730684v1) · Code: not checked
>
> PDF not committed.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: read (bioRxiv full text, first ~100k of 144k characters)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: an 860M+ parameter generative model pretrained on the "Human-Multi-Omics-Corpus" (~468M single-cell profiles, >425B tokens) that tokenises cis-regulatory elements, genes and proteins in one hierarchical vocabulary and generates with iterative diffusion-style masking/remasking. **(paper)**
- **Why it matters for the virtual cell**: it is the most direct large-scale instance of the Master's architecture (shared latent over ATAC+RNA, generative). It will be the "scale" baseline my context-focused hypothesis must beat or explain. **(my take)**
- **Main claim**: superior performance on single-omics representation, paired integration, unpaired alignment and cross-modal generation; generation order is not fixed, so it can "simulate information flow across omics layers". **(paper)**

## Research summary

- **Context (already known)**: single-cell foundation models are single-modality or pairwise, and RNA, ATAC and protein are measured in different amounts of data. **(paper)**
- **Gap (not known)**: whether one model can jointly represent and generate epigenome, transcriptome and proteome, in both paired and unpaired settings. **(paper)**
- **Hypothesis / what they try to determine**: a gene-anchored hierarchical tokenization with masked discrete diffusion supports representation, integration and any-to-any cross-modal generation. **(paper)**
- **Approach (how they investigated)**: pretrain an 860M-parameter encoder on HMOC-468M with span masking, reconstruction and alignment objectives; fine-tune with CLIP-style contrastive learning and optimal-transport pseudo-pairs; benchmark against modality-specific baselines. **(paper)**
- **Results (what they found)**: higher ARI/NMI than the best baselines on single-omics tasks, ATAC-to-RNA generation Pearson 0.798 vs 0.190 (EpiScanpy gene activity), RNA-to-protein 0.948 vs 0.392 (scTranslator). **(paper)**
- **Implications (what it means)**: cross-modal generation works at scale, but evidence is distributional and marker-level, not causal, so it does not yet test whether ATAC conditioning improves perturbation prediction. **(my take)**

## 2. Problem it solves

Existing scFMs are single-modality or pairwise; no model jointly represents and generates epigenome, transcriptome and proteome. **(paper)**

## 3. Method

- Hierarchical tokens: CREs (peaks) → genes → proteins in one framework. **(paper)**
- Masked discrete diffusion with remasking for any-to-any generation. **(paper)**
- Encoder: 24 layers, 20 heads, hidden 1,280. Each element (gene, protein, cCRE) is a triplet of two identity tokens and a value token (512 bins); cCREs are encoded by rank of genomic distance to the anchor gene (~1.35M cCREs). Sequence caps: 8,192 (ATAC), 4,096 (RNA), 1,024 (protein). **(paper)**
- Objectives (equal weight): span masked prediction, CLS signal reconstruction, cell-element alignment. Generation: iterative masked decoding from a fully masked target modality. Adaptation: CLIP-style contrastive fine-tuning; unpaired alignment via optimal-transport pseudo-pairs. **(paper)**

## 4. Data

Human-Multi-Omics-Corpus (HMOC-468M): ~67.9M transcriptomes, 11.4M epigenomes, 383.5M proteomes unimodal, plus ~2.4M paired ATAC-RNA and 2.9M paired RNA-protein cells; sources include CELLxGENE, CIMA, HDMA, SPDB, scMMO-atlas. **(paper)** Paired data are thus ~1% of the corpus. **(my take)** Data and weights described as under preparation.

## 5. Evaluation and results

- Single-omics clustering (HDMA RNA): zero-shot ARI +18.2% and NMI +5.3% over the strongest baseline; fine-tuned +22.9% / +9.5%. HDMA ATAC: zero-shot ARI +8.7%. **(paper)**
- Paired: PBMC 10x Multiome ARI/NMI 0.752/0.811 zero-shot; TEA-seq tri-omics 0.625/0.702 (~11% ARI over best unimodal ATAC). **(paper)**
- Unpaired ATAC-RNA: 97% label-transfer accuracy on brain Multiome; beats scGLUE and uniPort on ARI/NMI but scGLUE has higher iLISI. **(paper)**
- Generation: ATAC to RNA (PBMC) Pearson 0.798 vs 0.190; RNA to protein (Sacco2022) 0.948 vs 0.392, ARI of Leiden on generated protein 0.533 vs 0.585 real. Sequential ATAC to RNA to protein degrades in the second step. **(paper)**
- Perturbation prediction is not among evaluated tasks. **(paper)**

## 6. Critical analysis

- Limitations (paper): many zero-shot comparisons are against baselines fitted on the target data; RNA-to-protein Pearson computed on 0-1 rescaled outputs; generation assessed with distributional/marker metrics, not perturbation or causal validation; mainly human data; rank-based cCRE tokens may oversimplify distal enhancer logic and 3D chromatin.
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
