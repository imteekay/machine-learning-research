# E5.3: Does a niche model generalize to unseen tissues?

- **Roadmap phase**: [Phase 5](../../00-plan/roadmap.md#phase-5-spatial-and-tissue-level-modeling)
- **Status**: planned
- **Started / finished**: 2026-11-26 / —
- **Driving paper**: Novae (Blampey 2025)
- **Code package**: `novae_mini`

## Hypothesis

- **H**: A niche encoder trained on several tissues assigns niches and predicts neighborhood expression in a held-out tissue better than a model trained only on that tissue's limited data.
- **Null**: No transfer: the target-only model is equal or better.

## Why it moves the research forward

It decides whether one tissue world model is feasible or models must be disease-specific.

## Knowledge tested

- *Domain*: Niches shared across organs (lymphoid aggregates, vasculature, stroma) vs tissue-specific ones
- *Technical*: Graph SSL, leave-one-tissue-out, panel and technology shift, transfer matrices

## Design

- **Data**: Public 10x Xenium sections: breast, lung, colon, lymph node (~500k cells), restricted to shared panel genes
- **Models and baselines**: Mini Novae (GAT + SwAV-style prototypes). Baselines: cell-type composition kNN, target-only model, shuffled-coordinate control
- **Protocol**: run a `--quick` pilot on a 10% subsample, writing `results/E5.3/quick.json`. Then run the confirmatory run, writing `results/E5.3/confirmatory.json`.

## Metric and success criterion

Niche F1 against composition-derived labels, neighborhood-expression R², and a train × test transfer matrix. The thresholds above are fixed before running.

## Results

_Not run yet._

## Learnings ledger

Tag each bullet *new*, *confirms Ex.y* or *contradicts Ex.y*. Copy each bullet to [LEARNINGS.md](../LEARNINGS.md).

- *Biology*:
- *ML*:

## Conclusion

- **Verdict**:
- **Next step**:
