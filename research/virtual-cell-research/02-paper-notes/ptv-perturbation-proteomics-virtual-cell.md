# PTV-1: a perturbation proteomics-based virtual cell model

> *An operational perturbation proteomics-based virtual cell model*. Nature (2026). DOI: s41586-026-11001-9 · Earlier preprint: bioRxiv 10.1101/2025.02.07.637070 · Code: [guomics-lab/PTV-1](https://github.com/guomics-lab/PTV-1) (per secondary source, unverified)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 6](../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: skimmed (Nature login-walled; only search/secondary summaries)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a virtual cell built on protein-level perturbation data (ProteinTalks datasets), reported via secondary sources as ~38M protein abundance measurements over 16,311 perturbation proteomics samples (unverified against Nature).
- **Why it matters for the virtual cell**: proteins are the drug-acting layer; RNA-trained models can miss it.
- **Main claim**: abstract-only; the title claims an operational virtual cell. Results not retrieved.

## 2. Problem it solves

Most virtual cells model RNA; protein response to perturbation is the functionally relevant readout for drugs.

## 3-5. Method, data, results

Abstract-only: not retrieved. Preprint title: "A perturbation proteomics-based foundation model for virtual cell construction".

## 6. Critical analysis

- Limitations (paper): not retrieved.
- Weaknesses I see (my take): need to check baselines (mean/linear), held-out perturbation splits, and cell-line-only scope.
- What would change my mind: a mean-of-training-perturbations baseline matching it on held-out compounds.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md): protein layer for the multi-modal generative module; supports the "RNA is not enough" argument.
- Hypotheses worth testing:
  - **H-a (E3.x/E4.x)**: RNA-predicted drug responses correlate poorly with measured protein responses for post-transcriptionally regulated proteins; a protein-trained model closes the gap. Null: same.
  - **H-b**: leave-one-compound-out performance is within CI of a mean baseline.

## 8. Related work

HEx (spatial proteomics), CPA, scGen.

## Resources

- [Nature](https://www.nature.com/articles/s41586-026-11001-9) · [bioRxiv preprint](https://www.biorxiv.org/content/10.1101/2025.02.07.637070.full.pdf)
