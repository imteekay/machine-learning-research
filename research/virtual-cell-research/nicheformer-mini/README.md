# nicheformer-mini

A minimal, educational proof of concept for [E5.1](../04-experiments/E5.1-niche-context.md): *does niche context add information beyond cell type?* It is inspired by [Nicheformer](../02-paper-notes/nicheformer.md). It is not a reproduction.

- `src/nicheformer_mini/tissue.py`: a synthetic tumor section with ground truth (tumor/stroma/T-cell regions, an IFN program driven by nearby T cells, a hypoxic core, housekeeping genes, segmentation spillover).
- `src/nicheformer_mini/spatial.py`: k-NN graphs, neighborhood composition, neighborhood enrichment, spatial block folds.
- `src/nicheformer_mini/models.py`: ridge, cell-type mean, rank tokenization with context tokens, a 2-layer transformer (mini Nicheformer).
- `src/nicheformer_mini/experiment.py`: task A (expression from niche) and task B (niche composition from one cell).

```bash
uv sync
uv run pytest                              # ~2 s
uv run python scripts/eda_tissue.py        # results/eda/
uv run python scripts/run_e5_1.py --quick  # ~20 s smoke run
uv run python scripts/run_e5_1.py          # ~5 min, results/E5.1/
```

Data is synthetic only (no downloads). `data/` is gitignored for future public datasets.
