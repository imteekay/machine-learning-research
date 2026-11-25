# scGenePT: language gene embeddings added to scGPT for perturbation prediction

> Istrate A et al. (CZI; author list not verified). *scGenePT: Is language all you need for modeling single-cell perturbations?* bioRxiv (2024). DOI: [10.1101/2024.10.23.619972](https://doi.org/10.1101/2024.10.23.619972) · Code: see preprint

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. This note is **abstract-only**: the full text was not retrieved (paywall / rate limit), so no claim below comes from the methods or results sections.

- **Roadmap phase(s)**: [Phase 4](../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: extend scGPT's gene tokens with text embeddings of genes (NCBI gene card + UniProt summaries, as in GenePT; GO annotations) and fine-tune for perturbation prediction. **(paper)**
- **Why it matters for the virtual cell**: tests whether prior knowledge written in text can stand in for missing experimental data on unseen perturbations, the same role ATAC plays in the Master's hypothesis. **(my take)**
- **Main claim**: textual gene representations add complementary value to embeddings learned from expression alone; subcellular-location text helps most for single-gene perturbations and protein information most for combinations. **(paper)**

## 2. Problem it solves

Expression-only foundation models know a gene only through its co-expression. For a gene never perturbed in training, that may not be enough signal; curated knowledge (function, location, protein) may fill the gap. **(paper)**

## 3. Method

scGPT backbone; each gene token embedding is summed/combined with a language-model embedding of the gene's text description (GenePT-style OpenAI text embeddings of NCBI/UniProt summaries) and GO-annotation embeddings; fine-tuned on Perturb-seq with the GEARS-style task. Exact fusion and LLM version: abstract-only, not verified. **(paper / uncertain)**

## 4. Data

Perturb-seq benchmarks used in GEARS/scGPT (Norman combinatorial, Adamson); dataset list abstract-only, not verified. **(uncertain)**

## 5. Evaluation and results

abstract-only: reports gains over scGPT without language embeddings; no headline numbers recorded. Whether mean / linear baselines (Ahlmann-Eltze et al.) were included is unknown. **(my take)**

## 6. Critical analysis

- Limitations (paper): not retrieved (abstract-only).
- Weaknesses I see (my take): gains over scGPT are relative to a backbone already known to lose to simple linear baselines; text embeddings leak the literature (well-studied genes get better descriptions), which biases evaluation toward famous genes. **(my take)**
- What would change my mind: a gain that survives against a linear model on the same text embeddings, and that holds for poorly annotated genes.

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: a direct competitor to H1: if text priors already close the OOD gap for unseen knockouts, the ATAC contribution in E4.1 must be shown *on top of* text embeddings, not instead. **(my take)**
- Hypotheses worth testing (each could become an experiment in `../04-experiments/`): (1) *H*: in E4.1, adding GenePT embeddings to the RNA-only arm shrinks but does not eliminate the ATAC gain; falsified if RNA+text ≈ RNA+text+ATAC. (2) *H*: the text benefit decays with gene annotation depth (stratify by PubMed count, as in E4.2); falsified if the gain is flat.

## 8. Related work

GenePT, scGPT ([note](scgpt.md)), GEARS, Scouter ([note](scouter.md)), scLAMBDA ([note](sclambda.md)).

## Resources

- bioRxiv: https://www.biorxiv.org/content/10.1101/2024.10.23.619972v1
