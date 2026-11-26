# E6.3: Does closing the loop make discovery cheaper?

- **Roadmap phase**: [Phase 6](../../00-plan/roadmap.md#phase-6-world-models-closed-loop-and-clinical-translation)
- **Status**: planned
- **Started / finished**: 2026-11-26 / —
- **Driving paper**: Retrospective Bayesian optimization (e.g., GeneDisco)
- **Code package**: `loop_mini`

## Hypothesis

- **H**: An uncertainty-aware acquisition (ensemble UCB) finds the knockouts that revert a target signature with fewer experiments than random or diversity selection.
- **Null**: No reduction in experiments needed.

## Why it moves the research forward

It measures the proposal's 'accelerate discovery' promise and whether the model's uncertainty is calibrated.

## Knowledge tested

- *Domain*: Essential-gene complexes (ribosome, proteasome, mediator), what a disease-reversal signature is
- *Technical*: Bayesian experimental design, deep ensembles, calibration (ECE), retrospective simulation

## Design

- **Data**: Replogle 2022 K562 essential screen (~310k cells, ~2k knockouts). Target: knockouts whose pseudobulk Δ anti-correlates with a chosen program (e.g., an erythroid signature)
- **Models and baselines**: Acquisitions: random, diversity (k-center on GenePT/co-expression gene embeddings), greedy, ensemble UCB, ensemble Thompson sampling. 10 seeds, batches of 48
- **Protocol**: run a `--quick` pilot on a 10% subsample, writing `results/E6.3/quick.json`. Then run the confirmatory run, writing `results/E6.3/confirmatory.json`.

## Metric and success criterion

Hits vs experiments (area under the discovery curve), ECE of predicted effects. The thresholds above are fixed before running.

## Results

_Not run yet._

## Learnings ledger

Tag each bullet *new*, *confirms Ex.y* or *contradicts Ex.y*. Copy each bullet to [LEARNINGS.md](../LEARNINGS.md).

- *Biology*:
- *ML*:

## Conclusion

- **Verdict**:
- **Next step**:
