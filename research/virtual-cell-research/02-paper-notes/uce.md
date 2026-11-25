# UCE: universal cell embeddings across species via protein language model gene tokens

> Rosen Y, Roohani Y, Agrawal A, Samotorčan L, Tabula Sapiens Consortium, Quake SR, Leskovec J. *Universal cell embedding provides a foundation model for cell biology*. Nature (2026). DOI: [10.1038/s41586-026-10689-z](https://doi.org/10.1038/s41586-026-10689-z) · PMC: PMC13441871 · Code: [snap-stanford/UCE](https://github.com/snap-stanford/UCE) · [Video](https://www.youtube.com/watch?v=IWjKDYDJSt8)

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim. Only the abstract or secondary sources were read for this note (**abstract-only**); details beyond those are marked as missing rather than guessed.

- **Roadmap phase(s)**: [Phase 2](../00-plan/roadmap.md#phase-2-single-cell-foundation-models)
- **Status**: skimmed
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a self-supervised model that embeds any cell, from any tissue or species, into one latent space without labels, training or fine-tuning. **(paper)**
- **Why it matters for the virtual cell**: genes are represented via protein language model embeddings, so the model is not tied to a fixed gene vocabulary. That is what enables zero-shot embedding of species never seen in training, relevant to mouse→human translation. **(my take; PLM tokens per earlier preprint)**
- **Main claim**: the latent space captures biology despite noise, shows emergent structure (developmental lineages) and embeds unseen species. **(paper)**

## 2. Problem it solves

Atlases are fragmented by species, tissue and gene set; each new dataset usually needs integration or retraining. UCE aims for a single universal representation. **(paper)**

## 3. Method

Self-supervised transformer over cells; gene identity tokens derived from protein language model (ESM2) embeddings of the encoded protein, with genes sampled per cell weighted by expression and ordered by chromosome. **(my take: from the preprint, not re-checked in the Nature version)** Model size and objective details: abstract-only.

## 4. Data

Builds the **Integrated Mega-scale Atlas (IMA)**: 36M cells, >1,000 named cell types, hundreds of experiments, dozens of tissues, 8 species. **(paper)**

## 5. Evaluation and results

Emergent behaviour: developmental lineages, embedding of species not in training. **(paper)** Quantitative comparisons with scGPT/Geneformer/PCA: abstract-only.

## 6. Critical analysis

- Limitations (paper): not extracted.
- Weaknesses I see (my take): "zero-shot" embeddings from Geneformer/scGPT lost to HVG+PCA in the Kedzierska benchmark; UCE needs the same scrutiny. Protein-sequence gene tokens ignore regulatory (non-coding) context, which is exactly what my H1 says matters.
- What would change my mind: UCE beating HVG+PCA+Harmony on an out-of-pretraining tissue with donor-level splits.

## 7. Relevance to my research

- Relation to [proposal](../00-plan/research-proposal.md): UCE is one of the three models in **E2.1**. Its cross-species design supports translating mouse perturbation knowledge to human.
- Hypotheses:
  - H-U1 (**E2.1**): UCE zero-shot beats PCA on rare-cell-type macro-F1 in a tissue under-represented in IMA. Null: no gain beyond bootstrap CI.
  - H-U2: kNN transfer of mouse perturbation signatures into human cells in UCE space predicts human knockout direction (sign agreement on top DE genes) better than ortholog-mapped gene space. Null: no improvement.
  - H-U3 (**E2.3**): SAE features on UCE embeddings enrich Hallmark pathways but not TF→target edges, supporting adding ATAC (Phase 3).

## 8. Related work

TranscriptFormer, SCimilarity, scGPT, Geneformer, SATURN (same group).

## Resources

- [Paper](https://www.nature.com/articles/s41586-026-10689-z) · [Code](https://github.com/snap-stanford/UCE) · [Video](https://www.youtube.com/watch?v=IWjKDYDJSt8)
