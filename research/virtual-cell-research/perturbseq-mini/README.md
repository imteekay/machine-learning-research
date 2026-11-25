# perturbseq-mini

A minimal, educational Perturb-seq PoC for experiment [E0.2](../04-experiments/E0.2-tf-knockout-targets.md), built around Dixit et al. 2016 ([note](../02-paper-notes/perturb-seq-dixit-2016.md)).

- **Data**: the paper's own K562 TF knockout screen (GEO GSM2396858, `k562_tfs_7`, 327 MB) + CollecTRI TF→target edges from OmniPath.
- **Model**: MIMOSCA-style ridge regression, `z-scored expression ~ TF indicators + log depth covariates`, giving one effect vector β per TF.
- **Test**: mean |β| over a TF's CollecTRI targets vs 1000 expression-decile-matched random gene sets → permutation p → BH FDR. Baseline: pseudobulk Δ. Negative controls: shuffled labels, swapped prior.
- **Synthetic mode**: a toy GRN simulator with "dead" guides as ground truth (used by the tests).

## Layout

```
src/perturbseq_mini/  io, synthetic, preprocess, mimosca, enrichment, pipeline
scripts/eda_dixit.py  QC/MOI/depth/knockdown/prior-coverage figures -> results/E0.2/eda*
scripts/run_e0_2.py   main analysis -> results/E0.2/{full,quick,synthetic}/
tests/                pytest on the synthetic simulator (~1 s)
data/                 gitignored downloads + npz cache
```

## Run

```bash
uv sync
mkdir -p data/raw && cd data/raw
B=https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM2396nnn/GSM2396858/suppl/GSM2396858_k562_tfs_7
for s in .mtx.txt.gz _cbc_gbc_dict.csv.gz _cellnames.csv.gz _genenames.csv.gz; do curl -LO $B$s; done
curl -L "https://omnipathdb.org/interactions?datasets=collectri&genesymbols=yes&fields=type,curation_effort&license=academic" -o collectri.tsv
cd ../..
uv run pytest
uv run python -I scripts/eda_dixit.py
uv run python -I scripts/run_e0_2.py --quick   # ~6 s after the first load (~20 s)
uv run python -I scripts/run_e0_2.py           # full, ~20 s
uv run python -I scripts/run_e0_2.py --synthetic
```

The first load parses the MatrixMarket file and writes a ~1 GB npz cache in `data/`.
