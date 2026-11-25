# Atacformer: a fixed-region-vocabulary transformer for ATAC-seq with RNA–ATAC contrastive alignment (CRAFT)

> LeRoy N.J., Zheng G., Khoroshevskyi O., Campbell D.R. Jr., Zhang A., Sheffield N.C. *Atacformer: a transformer-based foundation model for analysis and interpretation of ATAC-seq data*. bioRxiv (2025). DOI: [10.1101/2025.11.03.685753](https://doi.org/10.1101/2025.11.03.685753) · [PMC12637716](https://pmc.ncbi.nlm.nih.gov/articles/PMC12637716/) · Code: [databio/gtars](https://github.com/databio/gtars), [databio/geniml](https://github.com/databio/geniml), HF `databio/atacformer-base-hg38`

Conventions: **(paper)** = stated in the paper; **(code)** = read from the repo; **(my take)** = my own inference, not the authors' claim.

- **Roadmap phase(s)**: [Phase 3](../00-plan/roadmap.md#phase-3-multi-omics-and-epigenetic-context)
- **Status**: read (PMC full text)
- **Read on**: 2026-11-25

## 1. TL;DR

- **What**: a lightweight transformer whose tokens come from a fixed universe of 890,704 genomic regions. It has no positional encoding and is pretrained with ELECTRA-style replaced-token detection on about 1.2M cells (about 10B tokens). **(paper)**
- **Why it matters for the virtual cell**: it is the opposite design choice to ChromFound (a vocabulary rather than coordinates), and it ships CRAFT, a CLIP-style ATAC to RNA aligner (Atacformer + Geneformer). That is exactly the "impute ATAC context for RNA-only Perturb-seq cells" module that E3.2 needs. **(my take)**
- **Main claim**: comparable or better clustering than ArchR/SnapATAC2/EpiAgent at much lower cost, plus region-level embeddings that are useful for interpretation (cryptic TSS discovery). **(paper)**

## 2. Problem it solves

Existing scATAC models embed cells but not *regions*, and they are slow (EpiAgent has about 1B parameters). Interpretation needs per-region embeddings, and atlas-scale use needs speed. **(paper)**

## 3. Method

- Tokenization: intersect each fragment or peak with the fixed region universe (Rust `gtars`, AIList/BITS). The cell is a bag of regions with no positional encoding. The context window is 8,192 tokens. **(paper)**
- Pretraining: about 45% of tokens are replaced with random ones, and the model classifies each token as replaced or original (ELECTRA). **(paper)**
- CRAFT: a dual encoder (Atacformer + Geneformer gf-12L-30M) with a CLIP loss, trained on about 106k multiome cells. A small MLP decodes RNA from the ATAC embedding. **(paper)**

## 4. Data

About 1.2M cells across 30 tissues from GEO, HCA, and 10x, uniformly reprocessed (CellRanger ATAC, SnapATAC2, MACS3). The bulk evaluation uses BEDbase (about 35k BED files). **(paper)**

## 5. Evaluation and results

- Fine-tuned for cell type: about +15% ARI across PBMC datasets. **(paper)**
- Fragment to embedding is about 80% faster end-to-end, and tokenization plus embedding is about 2.5x faster than EpiAgent. **(paper)**
- Bulk cell-line classification (XGBoost on embeddings): F1 0.85, about 86% accuracy over more than 275 lines. **(paper)**
- Cryptic TSSs show 6.33x (monocytes) and 6x (B cells) H3K4me3 enrichment, p < 0.001. **(paper)**
- Baselines include PCA, which is good. ChromFound was excluded (the authors say its code was not open). CRAFT's RNA imputation is shown qualitatively (marker plots), not with held-out correlation numbers. **(my take)**

## 6. Critical analysis

- Limitations (paper): no positional encoding, which hurts enhancer–promoter tasks. The model depends on the vocabulary and genome build. It was untested on noisy data. Clustering degrades when the context is truncated.
- Weaknesses I see (my take): a bag-of-regions model cannot learn distance-dependent regulation, which is the thing the Master's hypothesis needs. CRAFT's ATAC to RNA claim lacks a quantitative comparison against gene activity scores.
- What would change my mind: if CRAFT beats gene-activity kNN on FOSCTTM and held-out gene Pearson, it becomes the default E3.2 module.

## 7. Relevance to my research

- **Hypotheses**: CRAFT is a direct candidate for E3.2 (matching paired cells across modalities), which lets E4.1 use ATAC context for RNA-only screens.
- **Hypotheses worth testing**:
  - H-AF1 (E3.2): CRAFT recovers held-out multiome pairings with FOSCTTM below 0.1 and beats gene-activity kNN. *Null*: no better than gene activity.
  - H-AF2: because there is no positional encoding, Atacformer embeddings carry less enhancer–promoter information than ChromFound's. Test this on E3.1's stratified masked-gene task: the gain on genes with many distal peaks should be smaller. *Null*: equal gains.

## 8. Related work

[ChromFound](chromfound.md), [EpiAgent](epiagent.md), [EpiFoundation](epifoundation.md), scEmbed, Geneformer, scCLIP.

## Resources

- PMC full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC12637716/
- Tokenizers: https://github.com/databio/gtars · https://github.com/databio/geniml
