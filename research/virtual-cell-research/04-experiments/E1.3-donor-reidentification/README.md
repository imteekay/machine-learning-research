# E1.3: How identifiable are donors in your training data?

- **Roadmap phase**: [Phase 1](../../00-plan/roadmap.md#phase-1-single-cell-data-and-tooling)
- **Status**: planned
- **Started / finished**: 2026-11-26 / —
- **Driving paper**: Walker et al. 2024 Cell (private information leakage)
- **Code package**: `privacy_mini`

## Hypothesis

- **H**: Donor identity can be predicted from pseudobulk expression well above chance, driven by sex (XIST/chrY), HLA and eQTL genes. Removing those genes cuts donor-ID accuracy by ≥ 50% with a cell-type macro-F1 loss < 0.02.
- **Null**: Donors are not identifiable, or removing the genes destroys cell-type utility.

## Why it moves the research forward

It turns the privacy constraint for clinical data (Phase 6) into a measured trade-off and a checklist.

## Knowledge tested

- *Domain*: Genetic variation that shows up in RNA: sex chromosomes, HLA polymorphism, cis-eQTLs
- *Technical*: Linkage attacks, privacy–utility trade-off, donor-level splits

## Design

- **Data**: CELLxGENE Census PBMC slice, ≥ 200 donors, ~300k cells streamed. Linkage task: two disjoint cell halves per donor. **Report only aggregate metrics, never per-donor outputs**
- **Models and baselines**: Logistic regression / kNN linkage on pseudobulk, with all genes vs minus the sensitive set vs minus a random matched set. Cell-type classifier for utility
- **Protocol**: run a `--quick` pilot on a 10% subsample, writing `results/E1.3/quick.json`. Then run the confirmatory run, writing `results/E1.3/confirmatory.json`.

## Metric and success criterion

Top-1 linkage accuracy vs 1/N chance. Drop after removing sensitive vs random genes (bootstrap CI). Cell-type macro-F1 change. The thresholds above are fixed before running.

## Results

_Not run yet._

## Learnings ledger

Tag each bullet *new*, *confirms Ex.y* or *contradicts Ex.y*. Copy each bullet to [LEARNINGS.md](../LEARNINGS.md).

- *Biology*:
- *ML*:

## Conclusion

- **Verdict**:
- **Next step**:
