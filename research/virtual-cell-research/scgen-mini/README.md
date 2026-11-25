# scgen-mini

A minimal, educational scGen (Lotfollahi et al. 2019): a VAE whose latent δ (stim − ctrl), learned on some
cell types, is added to an **unseen** cell type's control cells to predict its IFN-β response.
Experiment write-up: [E4.5](../04-experiments/E4.5-scgen-unseen-cell-type.md) · paper note: [scGen](../02-paper-notes/scgen.md).

## Layout

- `src/scgen_mini/data.py`: GEO download list, Kang 2018 loader/preprocessing, synthetic toy with known truth.
- `src/scgen_mini/models.py`: control, gene-space shift, PCA arithmetic, mini scGen VAE, negative controls.
- `src/scgen_mini/metrics.py`: R² of means (scGen), Δ Pearson, Systema-style residuals, energy distance.
- `src/scgen_mini/experiment.py`: leave-one-cell-type-out loop.
- `scripts/eda_kang.py`, `scripts/run_e45.py`; `tests/` (synthetic, ~10 s); `results/E4.5/` (JSON + PNG).

## Run

```bash
uv sync
# data (~60 MB, public, into gitignored data/raw/)
mkdir -p data/raw && cd data/raw
B=https://ftp.ncbi.nlm.nih.gov/geo
for u in samples/GSM2560nnn/GSM2560248/suppl/GSM2560248_2.1.mtx.gz \
         samples/GSM2560nnn/GSM2560248/suppl/GSM2560248_barcodes.tsv.gz \
         samples/GSM2560nnn/GSM2560249/suppl/GSM2560249_2.2.mtx.gz \
         samples/GSM2560nnn/GSM2560249/suppl/GSM2560249_barcodes.tsv.gz \
         series/GSE96nnn/GSE96583/suppl/GSE96583_batch2.genes.tsv.gz \
         series/GSE96nnn/GSE96583/suppl/GSE96583_batch2.total.tsne.df.tsv.gz; do curl -sfLO $B/$u; done
cd ../..
uv run python -I scripts/eda_kang.py
uv run python -I scripts/run_e45.py --quick   # ~1 min
uv run python -I scripts/run_e45.py           # confirmatory, 3 types x 3 seeds, ~5-10 min CPU
uv run pytest
```

Without the files, `run_e45.py` falls back to a synthetic toy (shared + type-specific response) and says so.
pertpy's `kang_2018` figshare mirror returned 403 from this machine, so the loader reads the GEO originals.
