# get-mini

A minimal, educational take on **GET** (Fu et al. 2025, *Nature*): predict a gene's expression from the accessibility of the regulatory regions around it. It is used for experiment [E3.1: Is the ATAC information regulatory?](./README.md). Paper note: [GET](../../02-paper-notes/get-general-expression-transformer.md).

It is a toy, not a reproduction: no DNA sequence and no motifs, a 2-layer transformer, and ~150 PBMC metacells.

## Layout

```
src/get_mini/
  data.py        load a 10x Multiome filtered h5 (RNA + ATAC in one matrix)
  synthetic.py   synthetic multiome with known enhancer-driven vs housekeeping genes
  preprocess.py  log-CPM, PCA, k-means, metacells (joint RNA+ATAC), peak-gene linking, tau
  models.py      ridge with GCV; TinyGET (region tokens -> transformer -> attention pooling)
  experiment.py  E3.1 per-seed run + summary statistics
  pipeline.py    shared load + prepare
scripts/
  eda_multiome.py   EDA figures + json  -> results/eda/
  run_e3_1.py       experiment          -> results/E3.1[-quick|-synthetic]/
tests/              fast pytest on synthetic fixtures
```

## Run

```bash
uv sync
# data (39 MB + 78 MB, public 10x dataset; gitignored)
mkdir -p data && cd data
B=https://cf.10xgenomics.com/samples/cell-arc/2.0.0/pbmc_granulocyte_sorted_3k/pbmc_granulocyte_sorted_3k
curl -LO ${B}_filtered_feature_bc_matrix.h5
curl -LO ${B}_per_barcode_metrics.csv
cd ..
uv run python -I scripts/eda_multiome.py
uv run python -I scripts/run_e3_1.py --quick       # ~1.5 min, 1 seed, 500 genes
uv run python -I scripts/run_e3_1.py               # confirmatory: 3 seeds, 1000 genes
uv run python -I scripts/run_e3_1.py --synthetic   # positive control with known truth
uv run pytest
```

If the h5 file is missing, the scripts fall back to the synthetic generator and label their outputs accordingly.
