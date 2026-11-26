# scclip-mini

A minimal scCLIP-style RNA↔ATAC cell matcher (contrastive dual encoder) compared with gene-activity matching and Gromov–Wasserstein optimal transport. Experiment write-up: [E3.2](./README.md).

## Layout

- `src/scclip_mini/data.py`: loader and cache for the NeurIPS 2021 BMMC Multiome h5ad (GEO GSE194122: 69,249 cells, 4 sites, 10 donors, 22 cell types), the leave-one-site-out split with an unseen-donor evaluation pool, and a synthetic paired world.
- `src/scclip_mini/models.py`: RNA features (HVG → PCA 100), ATAC features (TF-IDF → SVD 100, first component dropped), `DualEncoder` + InfoNCE training, and (fused) Gromov–Wasserstein plans from POT. All of it is fit on the 3 training sites only.
- `src/scclip_mini/experiment.py`: per-cell FOSCTTM, top-k retrieval, cell-type agreement, the methods and controls, and ATAC imputation for RNA-only cells (R² on the top 10 LSI components).
- `scripts/run_e32.py`: synthetic, quick and confirmatory runs. Only aggregate metrics are written.
- `tests/` (~10 s): FOSCTTM on identity / random / tied scores; the split has only unseen donors; the contrastive model learns on a synthetic world while shuffled-pair and untrained controls stay at chance.

## Run

```bash
uv sync
mkdir -p data && cd data    # 2.9 GB download, 3.1 GB unpacked, gitignored
curl -sfLO https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz && gunzip GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad.gz
cd ..
uv run python -I scripts/run_e32.py --synthetic --epochs 15   # no download needed (the first real run builds data/bmmc_*.npz, ~10 min)
uv run python -I scripts/run_e32.py --quick --device mps
uv run python -I scripts/run_e32.py --device mps               # confirmatory: 4 held-out sites x 3 seeds
uv run pytest
```

## Paper → minimal architecture

| scCLIP component | Kept / cut / changed |
|---|---|
| Separate RNA and ATAC encoders, CLIP loss with a learnable temperature | kept (2-layer MLPs on PCA / LSI inputs, 128-d, symmetric InfoNCE) |
| Transformer encoders on gene / peak tokens | cut: PCA and LSI inputs |
| Pretraining on millions of cells | changed to the 3 training sites (about 52k cells) |
| Batch-aware sampling | cut |
| Gene-activity baseline | shipped gene-activity matrix, cosine on shared training HVGs |
| Labeled GW-OT (scOT / SCOT-style) | fused GW with a soft label cost from kNN classifiers fit on training cells |
