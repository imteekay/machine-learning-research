# privacy-mini

A minimal privacy audit of single-cell expression: how well can two disjoint cell sets from the same donor be linked, and what does removing genes cost in utility? Experiment write-up: [E1.3](./README.md).

## Layout

- `src/privacy_mini/data.py`: OneK1K loader (CELLxGENE Census, cached, donor IDs recoded to integers), the pre-registered gene sets, and a synthetic cohort with known genotype genes.
- `src/privacy_mini/experiment.py`: donor split, pseudobulk halves, linkage accuracy with bootstrap CI, matched-random gene removal, cell-type utility.
- `scripts/run_e13.py`: quick, confirmatory and synthetic runs. Only aggregate metrics are written.
- `tests/` (~10 s): the synthetic genotype genes are the most donor-specific; linkage drops when they are removed; the full run works end to end.

## Run

```bash
uv sync
uv run python -I scripts/run_e13.py --synthetic   # no download
uv run python -I scripts/run_e13.py --quick       # first run downloads ~120k cells from Census (~4 min), cached in data/
uv run python -I scripts/run_e13.py               # confirmatory: 3 seeds, ~1 min
uv run pytest
```
