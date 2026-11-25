# Tahoe-x1: perturbation-trained scFMs up to 3B parameters

> Tahoe Therapeutics. *Tahoe-x1: Scaling Perturbation-Trained Single-Cell Foundation Models to 3 Billion Parameters*. bioRxiv (2025). DOI: [10.1101/2025.10.23.683759](https://www.biorxiv.org/content/10.1101/2025.10.23.683759v1) · Weights: [tahoebio/Tahoe-x1](https://huggingface.co/tahoebio/Tahoe-x1)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Only the abstract or secondary sources were read for this note (**abstract-only**); details beyond those are marked as missing rather than guessed.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models) · [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: scGPT-style encoders in three sizes (70M, 1.3B, 3B; 32 layers at 3B) pretrained on up to 266M cells including the Tahoe-100M drug-perturbation atlas. **(preprint, via secondary sources)**
- **Why it matters for the virtual cell**: pretraining *with* perturbation data tests whether interventional data teaches causal structure that observational atlases cannot. **(my take)**
- **Main claim**: 3–30× higher compute efficiency than prior cell-state model implementations, with strong cancer-focused benchmarks (DepMap essentiality, MSigDB Hallmarks). **(preprint/company)**

## 2. Problem it solves

Observational atlases may not teach models how cells respond to intervention; cancer drug discovery needs perturbation-aware representations. **(my take)**

## 3. Method

scGPT-derived encoder, FSDP + mixed precision on 128 GPUs. **(secondary)** Objective details: abstract-only.

## 4. Data

Tahoe-100M (>100M cells, 50 cancer cell lines, >1,100 small molecules), Arc scBaseCount, CELLxGENE; up to 266M cells. **(secondary)**

## 5. Evaluation and results

DepMap essentiality, MSigDB Hallmark recovery, and perturbation tasks. **(secondary)** Numbers: abstract-only.

## 6. Critical analysis

- Limitations (paper): not extracted.
- Weaknesses I see (my take): cancer cell lines only for perturbations; gains could be cell-line identity memorization. Company-authored benchmarks.
- What would change my mind: gains on unseen cell lines × unseen drugs vs a mean-per-drug baseline.

## 7. Relevance to my research

- Relation to [proposal](../00-plan/research-proposal.md): a bridge between Phase 2 and **E4.1**; answers whether perturbation pretraining beats adding ATAC context.
- Hypotheses:
  - H-TX1 (**E4.1**): Tahoe-x1 embeddings reduce error on unseen genetic knockouts (Replogle) vs observational-only FM of similar size. Null: no gain (drug pretraining doesn't transfer to CRISPR).
  - H-TX2 (**E4.4**): Tahoe-x1's perturbation advantage on unseen cell lines disappears against a "mean response of the drug" baseline. Null: advantage persists.

## 8. Related work

scGPT, STATE (Arc), CellFM, PertEval-scFM.

## Resources

- [Preprint](https://www.biorxiv.org/content/10.1101/2025.10.23.683759v1) · [HF](https://huggingface.co/tahoebio/Tahoe-x1)
