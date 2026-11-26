# scvi-mini

A minimal, educational scVI (Lopez et al. 2018). One encoder and interchangeable observation models (Poisson, NB, ZINB, and a Gaussian on log1p(CP10k)), all scored in the **same count space**. Experiment write-up: [E0.1](./README.md).

## Layout

- `src/scvi_mini/data.py`: the 10x PBMC v3/v2 loaders, QC, HVG selection on log data (neutral to every likelihood), and a synthetic NB/ZINB world with known truth.
- `src/scvi_mini/likelihoods.py`: Poisson, NB and ZINB log-PMFs, plus a **discretised Gaussian**. The discretised Gaussian integrates the log-space density over each count's bin, so a density and a mass can be compared fairly.
- `src/scvi_mini/models.py`: the `MiniSCVI` VAE. `n_latent=0` turns it into the per-gene floor. `sample_counts` draws posterior-predictive samples.
- `src/scvi_mini/experiment.py`:
  - the 80/20 cell split
  - the six models: four likelihoods, `nb_floor`, and `nb_shuffled`, whose genes are permuted across cells
  - held-out log-likelihood per cell
  - posterior-predictive checks (PPC): variance and zero fraction against observed
  - paired bootstrap
- `scripts/eda_counts.py`: model-free zero-fraction check (Svensson 2020). `scripts/run_e01.py`: quick, confirmatory and synthetic runs.
- `tests/` (~15 s): every PMF sums to 1; ZINB reduces to NB as π → 0; NB beats Poisson in an NB world; ZINB wins in a zero-inflated world.

## Run

```bash
uv sync
mkdir -p data/raw && cd data/raw   # ~130 MB, public, gitignored
curl -sfLO https://cf.10xgenomics.com/samples/cell-exp/3.0.0/pbmc_10k_v3/pbmc_10k_v3_filtered_feature_bc_matrix.tar.gz
curl -sfLO https://cf.10xgenomics.com/samples/cell-exp/2.1.0/pbmc8k/pbmc8k_filtered_gene_bc_matrices.tar.gz
mkdir -p v3 v2 && tar xzf pbmc_10k_v3_filtered_feature_bc_matrix.tar.gz -C v3 && tar xzf pbmc8k_filtered_gene_bc_matrices.tar.gz -C v2
cd ../..
uv run python -I scripts/eda_counts.py
uv run python -I scripts/run_e01.py --quick   # ~15 s
uv run python -I scripts/run_e01.py           # confirmatory: 2 chemistries x 3 seeds x 6 models, ~30 min CPU
uv run pytest
```

Without the files, `run_e01.py` falls back to the synthetic worlds and says so.

## Paper → minimal architecture

| scVI component | Kept / cut / changed |
|---|---|
| Encoder on log1p counts, Gaussian q(z) | kept (1 hidden layer, 128 units, d = 10) |
| Softmax decoder × library size | kept, with the observed library instead of a learned one (scVI's `use_observed_lib_size`) |
| Gene-wise θ | kept |
| ZINB dropout head | kept, per cell × gene |
| Batch covariates | cut (one batch per dataset) |
| Library-size latent ℓ | cut |
| Gaussian-on-log baseline | added, with a discretised PMF for evaluation |
