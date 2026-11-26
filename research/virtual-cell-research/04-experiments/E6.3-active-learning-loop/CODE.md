# loop-mini

A minimal closed-loop experiment selector: a bootstrap-ensemble model plus acquisition functions, replayed retrospectively on a finished genome-scale-style Perturb-seq screen. Experiment write-up: [E6.3](./README.md).

## Layout

- `src/loop_mini/data.py`: streams the Replogle 2022 K562 essential screen (scPerturb h5ad, 310,385 cells), computes each knockdown's batch-matched effect on two pre-defined programs, split-half reliability, and the pre-experiment features (ESM2 protein embedding from E2.4, co-expression embedding from non-targeting cells only); a synthetic pool.
- `src/loop_mini/active.py`: `Ensemble` (8 MLPs trained jointly on bootstrap resamples), `k_center`, `acquire` (random, diversity, greedy, ucb, thompson, uncertainty, oracle), `calibration` (coverage and ECE of central intervals on the not-yet-revealed pool), `simulate`.
- `scripts/build_data.py`, `scripts/run_e63.py` (synthetic / quick / confirmatory).
- `tests/` (~30 s): the ensemble learns and is less certain far from data; k-center spreads its picks; calibration metric on a well-specified and an overconfident Gaussian; acquisition respects the revealed set and the oracle beats UCB beats random in a synthetic pool.

## Run

```bash
uv sync
mkdir -p data && cd data   # 1.55 GB, gitignored. Zenodo throttles each connection, so fetch the file in 8 parallel byte ranges and concatenate
# https://zenodo.org/records/7041849/files/ReplogleWeissman2022_K562_essential.h5ad   (scPerturb)
cd ..
uv run python -I scripts/build_data.py     # ~3 min; needs ../E2.4-cross-species-transfer/data/esm2_8m_human.npz (run E2.4's embed_proteins.py first)
uv run python -I scripts/run_e63.py --synthetic
uv run python -I scripts/run_e63.py --quick
uv run python -I scripts/run_e63.py
uv run pytest
```

## Paper → minimal design

| Idea in the literature (GeneDisco-style benchmarks, Bayesian experimental design) | Kept / changed |
|---|---|
| Batch acquisition with model uncertainty | kept: batches of 48, top-b by acquisition score (no batch diversity term) |
| Deep-ensemble epistemic uncertainty | kept: 8 MLPs on bootstrap resamples |
| Pre-trained gene embeddings as features | changed: a protein embedding (ESM2-8M) and a co-expression embedding built from non-targeting cells |
| Optimise a phenotype of interest | kept as two pre-defined program scores; hits are the top 5% |
| Real wet-lab rounds | replaced by replaying the finished screen |
