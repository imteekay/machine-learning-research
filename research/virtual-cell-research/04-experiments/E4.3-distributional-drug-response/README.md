# E4.3: Do distributional models capture responders vs non-responders to drugs?

- **Roadmap phase**: [Phase 4](../../00-plan/roadmap.md#phase-4-perturbation-modeling-and-causal-prediction)
- **Status**: planned
- **Started / finished**: 2026-11-26 / —
- **Driving paper**: CPA (Lotfollahi 2023); CellFlow / flow matching
- **Code package**: `flowpert_mini`

## Hypothesis

- **H**: On held-out drug × dose combinations, a conditional flow-matching model has lower E-distance than a CPA-style mean predictor with matched mean-Δ accuracy, and recovers responder fractions within CI.
- **Null**: Equal E-distance once the mean is matched.

## Why it moves the research forward

The resistant subpopulation is often what decides clinical outcome. This also covers E0.3 (what E-distance rewards) on real data.

## Knowledge tested

- *Domain*: Pharmacology: mechanism of action (MoA) classes (HDAC, kinase, epigenetic), dose–response, cell-line-specific sensitivity
- *Technical*: Conditional flow matching / OT, mode collapse, E-distance vs MSE, dose interpolation

## Design

- **Data**: sci-Plex3 (~650k cells, 188 drugs × 4 doses × A549 / K562 / MCF7). Splits: unseen dose and unseen drug within MoA
- **Models and baselines**: Control mean, linear dose model, CPA-mini (VAE + drug/dose embeddings), conditional flow matching. Shuffled-drug control
- **Protocol**: run a `--quick` pilot on a 10% subsample, writing `results/E4.3/quick.json`. Then run the confirmatory run, writing `results/E4.3/confirmatory.json`.

## Metric and success criterion

E-distance, Δ-Pearson top-50 DE, responder fraction (from a 2-component GMM on the response score), paired Wilcoxon across conditions. The thresholds above are fixed before running.

## Results

_Not run yet._

## Learnings ledger

Tag each bullet *new*, *confirms Ex.y* or *contradicts Ex.y*. Copy each bullet to [LEARNINGS.md](../LEARNINGS.md).

- *Biology*:
- *ML*:

## Conclusion

- **Verdict**:
- **Next step**:
