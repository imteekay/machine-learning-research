# scgpt-mini

A minimal, educational scGPT ([Cui et al. 2024](https://www.nature.com/articles/s41592-024-02201-0)) for experiment
[E2.1](./README.md): do frozen foundation-model cell embeddings beat HVG + PCA for
cell-type annotation? Paper note: [scgpt.md](../../02-paper-notes/scgpt.md).

Not a reproduction: a ~0.1M-parameter model, pretrained on 1.8k cells for a few minutes on CPU.

## What is in it

| File | Role |
|---|---|
| `src/scgpt_mini/data.py` | PBMC loaders (scanpy `pbmc3k_processed`, `pbmc68k_reduced`), per-cell quantile **value binning**, synthetic fallback generator |
| `src/scgpt_mini/model.py` | gene-id embedding + value-bin embedding → transformer encoder → `<cls>`; GEP head (masked token → value) and GEPC/MVC head (`<cls>` × gene → value) |
| `src/scgpt_mini/train.py` | masked-value pretraining |
| `src/scgpt_mini/evaluate.py` | HVG+PCA baseline, kNN / logistic probes, macro-F1, paired bootstrap CIs |
| `src/scgpt_mini/pipeline.py` | E2.1 protocol: pretrain on pbmc3k train cells → probe pbmc3k held-out (in-dist) and pbmc68k (transfer) |
| `scripts/eda_pbmc.py` | sparsity, library size, HVG, binning, label imbalance figures → `results/eda.json` |
| `scripts/run_e21.py` | main run (`--quick` for a 1-minute smoke test, `--synthetic` to skip downloads) → `results/results.json` + figures |

## Run

```bash
uv sync
uv run pytest                                   # 7 fast tests on synthetic data
uv run python -I scripts/eda_pbmc.py            # downloads ~30 MB to data/ (gitignored)
uv run python -I scripts/run_e21.py --quick     # ~1 min
uv run python -I scripts/run_e21.py             # 3 seeds, ~8 min on a laptop CPU
```

If the download fails, `run_e21.py` falls back to the synthetic generator and records `"data_source": "synthetic"` in the JSON.
