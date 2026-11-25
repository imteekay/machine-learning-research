# Nicheformer: a foundation model for single-cell and spatial omics

> Schaar, Tejada-Lapuerta, Palla, Gutgesell, Halle, Minaeva, Vornholz, Dony, Drummer, Bahrami & Theis. *Nicheformer: a foundation model for single-cell and spatial omics*. bioRxiv 2024; Nature Methods (2025). DOI: 10.1101/2024.04.15.589472 · Code: [theislab/nicheformer](https://github.com/theislab/nicheformer)
>
> PDF not committed. Author list, venue year and exact numbers below come from memory of the preprint and are marked **(uncertain)** where I could not re-check them.

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 5](../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: read (PoC in [`nicheformer-mini/`](../nicheformer-mini/), experiment [E5.1](../04-experiments/E5.1-niche-context.md))
- **Read on**: 2026-11-25
- **Why this paper**: picked by the citation ranking in `04-experiments/_paper-selection/` (Phase 0 PR, branch `vcell/E0.2-tf-knockout-targets`) as the top Phase 5 driving paper.

## 1. TL;DR

- **What**: a Geneformer-style transformer pretrained on ~110M cells **(paper, uncertain exact count)**, both dissociated scRNA-seq and image-based spatial transcriptomics (MERFISH, Xenium, CosMx, ISS), human and mouse.
- **Why it matters for the virtual cell**: it is the first large model to ask whether a *single cell's* expression encodes *where it lives*. That is the bridge from cell-level to tissue-level virtual cells.
- **Main claim**: joint pretraining on spatial + dissociated data gives embeddings that predict spatial labels (niche, region, neighborhood composition, neighborhood density) better than models trained on dissociated data only, and lets spatial labels be transferred onto dissociated cells **(paper)**.

## Research summary

- **Context (already known)**: scFMs pretrained on dissociated scRNA-seq never see spatial variation, and imaging spatial data measure only hundreds of genes. **(paper)**
- **Gap (not known)**: whether a single cell's expression encodes its tissue context and whether joint pretraining on spatial and dissociated data helps. **(paper)**
- **Hypothesis / what they try to determine**: pretraining on both modalities gives embeddings that predict spatial labels better than dissociated-only models. **(paper)**
- **Approach (how they investigated)**: Geneformer-style encoder with species, modality and technology tokens, pretrained on SpatialCorpus-110M; linear probing and fine-tuning on niche, region, neighbourhood composition and density. **(paper)**
- **Results (what they found)**: beats scGPT, Geneformer, UCE and PCA on spatial label and composition prediction, and models without spatial pretraining do worse; exact figures not re-checked. **(paper)**
- **Implications (what it means)**: spatial context can be inferred from one cell, but segmentation spillover, cell-type shortcuts and random splits could inflate gains without held-out-slide controls. **(my take)**

## 2. Problem it solves

Dissociated scRNA-seq has whole-transcriptome depth but loses position. Imaging spatial methods keep position but measure only a few hundred genes. scFMs pretrained only on dissociated data never see spatial variation, so they cannot be expected to encode it **(paper)**.

## 3. Method

- **Tokenization (paper)**: each cell → gene tokens sorted by expression *relative to the technology-specific mean* (rank encoding, as in Geneformer), truncated to 1,500 tokens **(uncertain)**. Normalizing by per-technology means is meant to remove panel and chemistry differences.
- **Context tokens (paper)**: prepended tokens for species, modality (dissociated vs spatial) and assay/technology. They let one model share weights across very different measurement regimes.
- **Gene vocabulary (paper)**: human and mouse orthologs mapped to shared tokens.
- **Backbone (paper)**: transformer encoder, 12 layers, ~49M parameters **(uncertain)**; self-supervised masked-token pretraining.
- **Spatial tasks (paper)**: linear probing and fine-tuning on niche label, region label, neighborhood composition (cell-type fractions in a radius) and neighborhood density, from *one cell's* embedding.

## 4. Data

SpatialCorpus-110M **(paper)**: public CELLxGENE-style dissociated data plus tens of millions of imaging-based spatial cells from many tissues. Spatial panels are small (hundreds of genes), so most spatial tokens are a subset of the dissociated vocabulary.

## 5. Evaluation and results

- Nicheformer beats scGPT/Geneformer/UCE embeddings and PCA on spatial label and composition prediction **(paper)**.
- Models pretrained without spatial data do worse on spatial tasks **(paper)**: the claimed value of spatial pretraining.
- Baselines: includes PCA and embedding baselines **(paper)**. I did not see a cell-type-mean baseline for neighborhood composition, which is the obvious one: a cell's type alone already predicts much of its neighborhood **(my take, uncertain whether it is in the supplement)**.

## 6. Critical analysis

- **Limitations (paper)**: imaging panels are small and differ by technology; spatial labels are partially defined from the same expression.
- **Weaknesses I see (my take)**:
  - *Segmentation spillover*: on imaging data, transcripts from neighbors leak into a cell. A model can then "predict its neighborhood" by reading neighbor markers inside the cell. That is a measurement artifact, not biology. My E5.1 toy shows spillover alone moves single-cell niche readout.
  - *Cell-type shortcut*: neighborhood composition is strongly determined by cell type (tumor cells live next to tumor cells). Gains should be reported over a cell-type-mean baseline.
  - *Spatial leakage*: random cell splits in a tissue put neighbors in train and test. Held-out regions/slides are needed.
- **What would change my mind**: gains over a cell-type mean baseline under held-out-slide evaluation, with a spillover-corrected or segmentation-free control.

## 7. Relevance to my research

- PhD H (Phase 5): spatially aware models beat isolated-cell models, especially at borders. Nicheformer is the *single-cell → niche* direction; E5.1 also tests the *niche → expression* direction.
- Hypotheses worth testing:
  - **E5.1** (done as a toy): niche features add R² beyond cell type, concentrated in niche-dependent programs; shuffled-neighbor control is null.
  - Does Nicheformer's niche-composition gain survive a cell-type-mean baseline and a spillover correction on real MERFISH/Xenium? (next)

## 8. Related work

- Geneformer (rank tokenization), scGPT, UCE (dissociated scFMs).
- [VirTues](./virtual-tissues-virtues.md) (spatial proteomics, image-level tokens).
- CellPLM, NicheCompass, GNN-based niche models (e.g., graph autoencoders over cell graphs).

## Resources

- Preprint: https://doi.org/10.1101/2024.04.15.589472
- Code: https://github.com/theislab/nicheformer
