# scTranslator: generative translation of single-cell transcriptomes into proteomes

> Liu L, Li W, Wang F, et al. *A pre-trained large generative model for translating single-cell transcriptome to proteome*. Nature Biomedical Engineering (2025). DOI: [10.1038/s41551-025-01528-z](https://doi.org/10.1038/s41551-025-01528-z) · preprint bioRxiv [2023.07.04.547619](https://www.biorxiv.org/content/10.1101/2023.07.04.547619) · Code: [WaveoffBioMed/scTranslator](https://github.com/WaveoffBioMed/scTranslator)
>
> PDF not committed (journal version paywalled).

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: skimmed (abstract-only: journal full text not accessible; numbers below are from the 2023 preprint via secondary summaries)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: an encoder-decoder transformer (Performer-style FAVOR+ attention, non-autoregressive decoding) pretrained first on paired bulk RNA/protein, then on paired single-cell data, that generates a cell's protein abundances from its transcriptome. **(paper)**
- **Why it matters for the virtual cell**: drugs act on proteins, and RNA-protein correlation is modest; a virtual cell that stops at RNA predicts the wrong layer for many interventions. scTranslator is a cheap "protein head" for RNA-only perturbation data. **(my take)**
- **Main claim**: beats cTP-net, sciPENN, Seurat, BABEL and others on protein inference, especially few-shot (reported ~80-95% of full-data performance after fine-tuning on ~20 cells), and supports downstream tasks (gene/protein interaction inference, perturbation, clustering, batch correction). **(paper, preprint; not verified against journal version)**

## 2. Problem it solves

CITE-seq-style protein measurements are expensive and panel-limited; most single-cell data are RNA only. Inferring the proteome from RNA unlocks protein-level analysis on existing atlases and screens. **(paper)**

## 3. Method

- Transformer encoder over RNA tokens, decoder that emits protein values for queried protein IDs in parallel (non-autoregressive), FAVOR+ linear attention for long gene sets. **(paper)**
- Two-stage pretraining: bulk paired RNA/protein (TCGA/CPTAC-like pan-cancer data), then paired single-cell data; fine-tuning for a new panel. **(paper; exact source split abstract-only)**
- Gene/protein position encoding by ID, so arbitrary panels can be requested. **(paper)**

## 4. Data

Preprint: 31 cancer types, 18,227 patient-level bulk samples, 239,797 single cells, 76 datasets in total. **(paper, preprint)** Specific single-cell benchmark sets: abstract-only, not checked.

## 5. Evaluation and results

- Task: predict protein abundance for held-out cells/datasets; metrics include Pearson correlation and error per protein. Baselines: nine frameworks incl. cTP-net, sciPENN, Seurat, BABEL. **(paper)**
- Headline: wins on five datasets in few-shot tests; ~80-95% of full performance from ~20 fine-tuning cells. **(paper, preprint; secondary source)**
- Missing in what I saw: a "copy RNA of the coding gene" baseline and a mean-per-protein baseline. **(my take)**

## 6. Critical analysis

- Limitations (paper): abstract-only; not extracted.
- Weaknesses I see (my take): RNA-to-surface-protein mapping is largely a cell-type lookup; high correlation across cells can come from cell-type identity rather than within-type variation. The perturbation use case (does it predict protein *changes*, not levels?) is the hard part and is unclear from the abstract.
- What would change my mind: within-cell-type correlation well above a cell-type-mean baseline, and correct protein fold-changes on a held-out perturbation dataset (e.g., ECCITE-seq).

## 7. Relevance to my research

- How it relates to the [research proposal](../00-plan/research-proposal.md) hypotheses: the Master's model predicts RNA (+ATAC); scTranslator is an off-the-shelf decoder that would let E4.1 outputs be read at the protein level, which is what a drug-discovery use case needs. It is also an analogue of the E3.2 "impute the missing modality" module, but for protein. **(my take)**
- Hypotheses worth testing:
  - **H-sct1**: scTranslator's protein predictions beat a cell-type-mean baseline *within* cell types (Pearson on residuals). *Null*: within-type correlation ≈ 0, i.e. it is a classifier in disguise. Design: CITE-seq PBMC, stratify by annotation.
  - **H-sct2**: feeding RNA predicted by an E4.1 model through scTranslator yields protein fold-changes that agree in sign with measured ECCITE-seq/Perturb-CITE-seq protein changes above chance. *Null*: sign agreement ≈ 50%.

## 8. Related work

cTP-net, sciPENN, totalVI, BABEL (RNA↔ATAC translation), [CAPTAIN](captain.md) (RNA+protein foundation model), [HoloCell](holocell.md), [scGPT](scgpt.md).

## Resources

- Journal: https://doi.org/10.1038/s41551-025-01528-z
- Preprint: https://www.biorxiv.org/content/10.1101/2023.07.04.547619
- Code: https://github.com/WaveoffBioMed/scTranslator
