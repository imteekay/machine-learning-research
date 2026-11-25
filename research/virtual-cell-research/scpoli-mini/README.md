# scpoli-mini

A minimal, educational re-implementation of the core ideas of **scPoli** (De Donno et al. 2023, Nature Methods): a conditional VAE with **learned condition (sample/study) embeddings** and a **cell-type prototype loss**. It drives experiment [E1.1: Integration without erasing disease](../04-experiments/E1.1-integration-without-erasing-disease.md). Paper note: [02-paper-notes/scpoli.md](../02-paper-notes/scpoli.md).

Not a reproduction: no query-time surgery, no unlabelled prototypes, no MMD, tiny networks.

## Layout

- `src/scpoli_mini/simulate.py`: synthetic multi-study PBMC-like counts with a known study effect and a known disease (ISG) program.
- `src/scpoli_mini/data.py`: normalisation, HVGs, ISG module score, Kang 2018 loader.
- `src/scpoli_mini/models.py`: `CondVAE` (`mode="onehot"` = scVI-like, `mode="embed"` = mini scPoli), NB loss, prototype loss, PCA baseline.
- `src/scpoli_mini/metrics.py`: variance explained by batch / cell type, kNN batch mixing, kNN cell-type accuracy, held-out-group disease AUROC, held-out ISG-score Spearman.
- `scripts/eda_e1_1.py`, `scripts/run_e1_1.py`: EDA figures and the experiment. Outputs go to `results/E1.1/`.

## Run

```bash
uv sync
# optional real data (38 MB, gitignored):
curl -o data/kang_2018.h5ad https://exampledata.scverse.org/pertpy/kang_2018.h5ad
uv run pytest
uv run python -I scripts/eda_e1_1.py
uv run python -I scripts/run_e1_1.py --quick   # ~2 min CPU; drop --quick for the full run (~8 min)
```

Without `data/kang_2018.h5ad` the scripts run on synthetic data only.
