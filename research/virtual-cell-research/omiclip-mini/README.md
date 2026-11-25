# omiclip-mini

A minimal, educational re-implementation of the idea behind **OmiCLIP / Loki** (Chen et al., Nature Methods 2025): align H&E image patches with spatial-transcriptomics spots in one CLIP space, where each spot's expression is written as a "gene sentence", then predict expression from histology by retrieval. Used for experiment [E6.2](../04-experiments/E6.2-histology-to-expression.md). Paper note: [omiclip-loki](../02-paper-notes/omiclip-loki.md).

Everything runs on **synthetic data with known truth** (no downloads): 24 synthetic patients, 150 spots each, 32×32 H&E-like tiles drawn from the same cells that generate the spot's counts.

```
src/omiclip_mini/
  simulate.py   tissue world: cells -> H&E tile + NB counts; gene sentences; biomarker scores
  model.py      CNN image encoder, gene-sentence encoder, InfoNCE training, kNN PredEx, baselines
  evaluate.py   patient-level Spearman (+bootstrap CI), gene Pearson, retrieval top-k, variance split
scripts/
  eda_e6_2.py   EDA figures + results/E6.2/eda.json
  run_e6_2.py   full experiment (3 seeds x 5 patient folds); --quick for a ~20 s smoke run
tests/          pytest, ~5 s
results/E6.2/   JSON + PNG figures
```

```bash
uv sync
uv run pytest
uv run python scripts/eda_e6_2.py
uv run python scripts/run_e6_2.py --quick   # smoke
uv run python scripts/run_e6_2.py           # confirmatory, seeds 0 1 2 (CPU, a few minutes)
```
