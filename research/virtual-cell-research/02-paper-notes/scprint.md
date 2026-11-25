# scPRINT and scPRINT-2: foundation models designed for gene network inference

> Kalfon J, Samaran J, Peyré G, Cantini L. *scPRINT: pre-training on 50 million cells allows robust gene network predictions*. Nature Communications (2025). DOI: [10.1038/s41467-025-58699-1](https://doi.org/10.1038/s41467-025-58699-1) · PMC12003772 · Code: [cantinilab/scPRINT](https://github.com/cantinilab/scPRINT). Follow-up: Kalfon J, Peyré G, Cantini L. *scPRINT-2: Towards the next-generation of cell foundation models and benchmarks*. bioRxiv (2025). DOI: [10.64898/2025.12.11.693702](https://www.biorxiv.org/content/10.64898/2025.12.11.693702v1) · [docs](https://cantinilab.github.io/scPRINT-2/model)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Only the abstract or secondary sources were read for this note (**abstract-only**); details beyond those are marked as missing rather than guessed.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a large cell model pretrained on >50M CELLxGENE cells, whose pretraining tasks and architecture target gene network inference from attention. scPRINT-2 scales to ~350M cells over 16 species and adds counterfactual expression prediction. **(paper; scPRINT-2 from blog/PyPI)**
- **Why it matters for the virtual cell**: it is the clearest attempt at the "Explain" capability: extracting regulatory networks from an FM rather than only embeddings. **(my take)**
- **Main claim**: better GRN inference than SOTA on atlas-level benchmarks, plus competitive zero-shot denoising, batch correction and label prediction. **(paper)**

## 2. Problem it solves

FMs learn something about gene–gene relations, but whether it is usable as a regulatory network is unclear. scPRINT designs pretraining to make GRNs extractable and benchmarks them. **(paper)**

## 3. Method

Transformer with "innovative pretraining tasks" (from memory: denoising, bottleneck learning, label prediction) and gene tokens from protein embeddings; GRNs read from attention heads. **(my take; abstract-only, not verified)**

## 4. Data

scPRINT: >50M cells from CELLxGENE. **(paper)** scPRINT-2: ~350M cells, ~25 TB, 16 species. **(blog/PyPI, not paper-verified)**

## 5. Evaluation and results

GRN inference: superior to SOTA on authors' atlas-level benchmarks; zero-shot denoising, batch correction, label prediction competitive. Case study on benign prostatic hyperplasia links ion exchange, senescence and chronic inflammation. **(paper)** Numbers and which ground truths (ChIP, perturbation, literature): abstract-only.

## 6. Critical analysis

- Limitations (paper): not extracted.
- Weaknesses I see (my take): GRN benchmarks are notoriously weak (ChIP/literature ground truth incomplete, co-expression baselines strong). Attention ≠ causal regulation.
- What would change my mind: attention-derived edges predicting Perturb-seq knockout targets better than GENIE3 and correlation.

## 7. Relevance to my research

- Relation to [proposal](../00-plan/research-proposal.md): the proposal lists GRNs as a required concept; scPRINT is the natural FM baseline for **E2.3** (TF→target recovery) and contrasts with ATAC-derived regulatory info in **E3.1**/**E3.3**.
- Hypotheses:
  - H-SP1 (**E2.3**): scPRINT attention edges beat co-expression on CollecTRI AUPRC but not on edges validated by knockout effects in Perturb-seq (**E0.2** targets). Null: no difference in either.
  - H-SP2: an scPRINT GRN prior plus ATAC peak-gene links beats either alone for predicting which genes move after a TF knockout (**E3.3**/**E4.2**). Null: no additive gain.

## 8. Related work

RegFormer, GENIE3, SCENIC+, scGPT, Geneformer.

## Resources

- [scPRINT paper](https://www.nature.com/articles/s41467-025-58699-1) · [scPRINT code](https://github.com/cantinilab/scPRINT) · [scPRINT-2 preprint](https://www.biorxiv.org/content/10.64898/2025.12.11.693702v1) · [scPRINT-2 PyPI](https://pypi.org/project/scprint2/) · [Lamin blog](https://blog.lamin.ai/scprint.md)
