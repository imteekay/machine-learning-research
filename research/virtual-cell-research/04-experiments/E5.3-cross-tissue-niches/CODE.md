# novae-mini

A minimal Novae-style spatial niche encoder (GraphSAGE + neighbour contrast), tested for transfer to an unseen tissue and to an unseen animal and cutting plane. Experiment write-up: [E5.3](./README.md).

## Layout

- `src/novae_mini/data.py`: loader and cache for the whole-mouse-brain MERFISH atlas from CELLxGENE Discover (animal 2 coronal: 1.9M cells, 66 sections; animal 4 sagittal: 215k cells, 3 sections), a per-section k-NN graph in micrometres, and a synthetic brain with known niche rules.
- `src/novae_mini/models.py`: gene PCA (fit on training cells only), the two-hop `NicheEncoder`, contrastive training (positive = one of the 5 nearest neighbours, negatives = other cells of the same section), a shuffled-graph control.
- `src/novae_mini/experiment.py`: the held-out evaluation set (cells in niches with at least 1,000 cells in the tissue), NMI / ARI of k-means clusters against the fine CCF regions, the baselines, and `run_model`.
- `scripts/run_e53.py`: synthetic / quick / confirmatory. Only aggregate metrics are written.
- `tests/` (~30 s): the graph stays in its section, NMI on perfect and random clusters, encoder shape and normalisation, and a synthetic brain where transfer beats expression PCA and the shuffled-graph control.

## Run

```bash
uv sync
mkdir -p data && cd data    # 1.3 GB, gitignored; ids from the CELLxGENE curation API (dataset_version_id)
curl -sfLO https://datasets.cellxgene.cziscience.com/979c38f6-63ee-48d0-b593-aaf2ea8b7152.h5ad   # WB_MERFISH_animal2_coronal
curl -sfLO https://datasets.cellxgene.cziscience.com/4b5c682f-4b5d-4dcb-81c1-606fef2b601e.h5ad   # WB_MERFISH_animal4_sagittal
cd ..
uv run python -I scripts/run_e53.py --synthetic --steps 300 --seeds 1
uv run python -I scripts/run_e53.py --quick       # first run builds data/brain_*.npz (~30 s)
uv run python -I scripts/run_e53.py               # confirmatory: 10 tissues x 3 seeds
uv run pytest
```

## Paper → minimal architecture

| Novae component | Kept / cut / changed |
|---|---|
| Graph attention encoder on a spatial cell graph | changed to a 2-layer GraphSAGE (mean aggregation) over the 10 nearest cells |
| Pretrained on about 30M cells, 18 tissues | changed to 360k cells of 9 mouse brain regions |
| Gene tokens from a shared gene-embedding table | changed to a per-model PCA 50 (so the panel must match; here every dataset has the same 1,120 genes) |
| SwAV-style prototype clustering with swapped assignments between a cell and its neighbours | changed to InfoNCE between a cell and one of its 5 nearest neighbours |
| Zero-shot niche assignment at several hierarchy levels | kept as zero-shot k-means on the embedding, scored against annotated regions |
